"""
app/graph/workflow.py

LangGraph Workflow Definition — P3 Production Version
------------------------------------------------------
Defines the complete review pipeline as a stateful directed graph.

Graph Flow
----------
START
  → fetch_diff_node
  → analyze_code_node
  → reflect_node          (×1 via should_reflect — quota-conscious)
  → lint_node
  → refactor_node         (conditional — only if lint fails)
  → validator_node        (conditional — only if refactor runs)
  → [PAUSE: interrupt_before hitl_node]
  → hitl_node             (reads human_decision injected via Command(resume=))
  → verdict_node
  → END

HITL Design
-----------
interrupt_before=["hitl_node"] is the ONLY correct mechanism for pausing.
It tells LangGraph to checkpoint state and raise GraphInterrupt BEFORE
hitl_node executes. The node itself is only executed on resume.

On first ainvoke/astream:
    Graph runs fetch → analyze → reflect → lint → [refactor] → PAUSE
    GraphInterrupt is raised — caught by review_service.py
    Review status set to pending_hitl

On resume via Command(resume="approved" | "rejected"):
    Graph resumes from checkpoint
    hitl_node executes — reads decision from interrupt() return value
    verdict_node produces final output

Error Routing
-------------
check_error() routes to hitl_node immediately on upstream failure.
This ensures a human always gets to inspect the result even on errors.

Routing Controllers
-------------------
check_error()         — after analyze: error → hitl, ok → reflect
should_reflect()      — after reflect: max 1 pass (quota-conscious)
should_lint_refactor()— after lint: passed → hitl, failed → refactor
should_refactor()     — after validator: passed → hitl, failed → loop (max 2)

Singleton Pattern
-----------------
get_review_graph() returns a cached instance.
review_graph module-level alias is required for review_service.py import.
MemorySaver is used for local development — swap for PostgresSaver in P6.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.graph.state import ReviewState
from app.graph.nodes import (
    fetch_diff_node,
    analyze_code_node,
    reflect_node,
    lint_node,
    refactor_node,
    validator_node,
    verdict_node,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Routing Controllers ───────────────────────────────────────────────────────

def check_error(state: ReviewState) -> str:
    """
    Route after analyze_code_node.

    If an upstream error occurred (GitHub fetch failure or hard LLM failure),
    skip reflection and route directly to hitl_node so a human can inspect.

    Returns
    -------
    str
        "hitl_node"   — on error
        "reflect_node" — normal path
    """
    if state.get("error"):
        logger.error(
            "[workflow] check_error: upstream error detected — "
            "reason=%s, routing to hitl_node",
            state.get("error_reason", "unknown"),
        )
        return "hitl_node"

    logger.debug("[workflow] check_error: no error — routing to reflect_node")
    return "reflect_node"


def should_reflect(state: ReviewState) -> str:
    """
    Control the reflection loop after reflect_node.

    Limited to 1 reflection pass to conserve free-tier LLM quota.
    Increase the threshold here if billing is enabled.

    Returns
    -------
    str
        "reflect_node" — if reflection_count < 1
        "lint_node"    — when reflection is complete
    """
    count = state.get("reflection_count", 0)

    if count < 1:
        logger.info(
            "[workflow] should_reflect: pass %d/1 — routing to reflect_node",
            count + 1,
        )
        return "reflect_node"

    logger.info(
        "[workflow] should_reflect: reflection complete (count=%d) "
        "— routing to lint_node",
        count,
    )
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    """
    Route after lint_node.

    If lint passed (or was skipped), proceed directly to hitl_node.
    If lint failed, route to refactor_node for corrective patch generation.

    Returns
    -------
    str
        "hitl_node"    — lint passed or skipped
        "refactor_node" — lint failed
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_lint_refactor: error detected — "
            "routing to hitl_node"
        )
        return "hitl_node"

    if state.get("lint_passed", True):
        logger.info(
            "[workflow] should_lint_refactor: lint passed — "
            "routing to hitl_node"
        )
        return "hitl_node"

    logger.warning(
        "[workflow] should_lint_refactor: lint failed — "
        "routing to refactor_node"
    )
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    """
    Control the refactor/validation loop after validator_node.

    Max 2 refactor attempts. After max attempts, proceeds to hitl_node
    regardless of validation result — human reviews the partial output.

    Returns
    -------
    str
        "hitl_node"    — validation passed, or max attempts reached
        "refactor_node" — validation failed and attempts remaining
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_refactor: error detected — "
            "routing to hitl_node"
        )
        return "hitl_node"

    count = state.get("refactor_count", 0)
    passed = state.get("validation_passed", False)

    if passed:
        logger.info(
            "[workflow] should_refactor: validation passed — "
            "routing to hitl_node"
        )
        return "hitl_node"

    if count < 2:
        logger.warning(
            "[workflow] should_refactor: validation failed — "
            "retrying refactor (attempt %d/2)",
            count + 1,
        )
        return "refactor_node"

    logger.warning(
        "[workflow] should_refactor: max refactor attempts reached (%d) — "
        "routing to hitl_node regardless of validation result",
        count,
    )
    return "hitl_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-In-The-Loop gate node.

    This node is NEVER executed on the first graph run because
    interrupt_before=["hitl_node"] pauses the graph BEFORE this node runs.

    On resume (via Command(resume=decision)):
        interrupt() returns the decision value injected by the caller.
        The node writes human_decision to state and returns.
        Graph continues to verdict_node.

    Guard Clause
    ------------
    If human_decision is already set (e.g. graph resumed a second time),
    the node returns immediately without calling interrupt() again.
    This prevents double-interrupt on retry scenarios.

    Parameters
    ----------
    state : ReviewState
        Current graph state at resume time.

    Returns
    -------
    dict
        {"human_decision": "approved" | "rejected"}
        or {} if already decided.
    """
    pr_number = state.get("pr_number")

    logger.info(
        "[hitl_node] Executing for PR #%s — "
        "this means graph was resumed via Command(resume=...)",
        pr_number,
    )

    # Guard — already decided on a previous resume
    if state.get("human_decision") is not None:
        logger.info(
            "[hitl_node] human_decision already set to '%s' — "
            "skipping interrupt(), returning empty dict",
            state["human_decision"],
        )
        return {}

    logger.info(
        "[hitl_node] Calling interrupt() — "
        "decision will be injected by review_service.decide_review()"
    )

    decision = interrupt(
        "Review paused. Awaiting human decision: approve / reject."
    )

    logger.info(
        "[hitl_node] Resumed with decision='%s' for PR #%s",
        decision, pr_number,
    )

    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the P3 LangGraph StateGraph.

    Compilation Options
    -------------------
    checkpointer=MemorySaver()
        Required for interrupt_before to work — graph state must be
        persisted so it can be resumed. MemorySaver stores in-process.
        For production (P6), replace with AsyncPostgresSaver.

    interrupt_before=["hitl_node"]
        CRITICAL — this is what makes HITL work.
        Tells LangGraph to checkpoint state and raise GraphInterrupt
        BEFORE hitl_node executes on the first run.
        Without this, interrupt() inside the node has no effect.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph application ready for ainvoke / astream.

    Raises
    ------
    Exception
        Re-raised if graph compilation fails — startup should fail loudly.
    """
    try:
        logger.info("[workflow] Building LangGraph workflow")

        builder = StateGraph(ReviewState)

        # ── Register nodes ────────────────────────────────────────────────────
        builder.add_node("fetch_diff_node",    fetch_diff_node)
        builder.add_node("analyze_code_node",  analyze_code_node)
        builder.add_node("reflect_node",       reflect_node)
        builder.add_node("lint_node",          lint_node)
        builder.add_node("refactor_node",      refactor_node)
        builder.add_node("validator_node",     validator_node)
        builder.add_node("hitl_node",          hitl_node)
        builder.add_node("verdict_node",       verdict_node)

        # ── Entry point ───────────────────────────────────────────────────────
        builder.set_entry_point("fetch_diff_node")

        # ── Core flow ─────────────────────────────────────────────────────────
        builder.add_edge("fetch_diff_node", "analyze_code_node")

        # After analysis: error → hitl, ok → reflect
        builder.add_conditional_edges(
            "analyze_code_node",
            check_error,
            {
                "reflect_node": "reflect_node",
                "hitl_node":    "hitl_node",
            },
        )

        # Reflection loop: max 1 pass
        builder.add_conditional_edges(
            "reflect_node",
            should_reflect,
            {
                "reflect_node": "reflect_node",
                "lint_node":    "lint_node",
            },
        )

        # Lint result routing
        builder.add_conditional_edges(
            "lint_node",
            should_lint_refactor,
            {
                "refactor_node": "refactor_node",
                "hitl_node":     "hitl_node",
            },
        )

        # Refactor → Validation
        builder.add_edge("refactor_node", "validator_node")

        # Validation loop: max 2 refactor attempts
        builder.add_conditional_edges(
            "validator_node",
            should_refactor,
            {
                "refactor_node": "refactor_node",
                "hitl_node":     "hitl_node",
            },
        )

        # HITL → Verdict → END
        builder.add_edge("hitl_node",   "verdict_node")
        builder.add_edge("verdict_node", END)

        # ── Compile ───────────────────────────────────────────────────────────
        memory = MemorySaver()

        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_node"],
            # ↑ CRITICAL — without this, interrupt() has no effect.
            # Graph pauses BEFORE hitl_node on first run.
            # hitl_node only executes after Command(resume=decision).
        )

        logger.info("[workflow] LangGraph workflow compiled successfully")
        return graph

    except Exception:
        logger.exception("[workflow] Graph compilation failed")
        raise


# ── Singleton Management ──────────────────────────────────────────────────────

_review_graph = None


def get_review_graph():
    """
    Lazily initialize and return the singleton graph instance.

    The graph is expensive to build (StateGraph compilation + MemorySaver).
    Built once on first request, reused for all subsequent requests.

    Returns
    -------
    CompiledStateGraph
        The singleton compiled graph instance.
    """
    global _review_graph

    if _review_graph is None:
        logger.info("[workflow] Initializing graph lazily")
        _review_graph = build_graph()

    return _review_graph


# Module-level alias — required for: from app.graph.workflow import review_graph
review_graph = get_review_graph()