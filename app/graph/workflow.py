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
  → lint_node             (reflection skipped — REFLECTION_PASSES=0)
  → refactor_node         (conditional — only if lint fails)
  → validator_node        (conditional — only if refactor runs)
  → [PAUSE: interrupt_before hitl_node]
  → hitl_node             (reads human_decision injected via Command(resume=))
  → verdict_node
  → END

Reflection Loop
---------------
REFLECTION_PASSES controls how many self-reflection passes run after
analyze_code_node. Set to 0 for development to conserve free-tier quota
(1 LLM call per review total). Set to 1 or 2 for production.

    REFLECTION_PASSES = 0  → analyze → lint          (1 LLM call)
    REFLECTION_PASSES = 1  → analyze → reflect → lint (2 LLM calls)
    REFLECTION_PASSES = 2  → analyze → reflect × 2 → lint (3 LLM calls)

HITL Design
-----------
interrupt_before=["hitl_node"] is the ONLY correct mechanism for pausing.
It tells LangGraph to checkpoint state and raise GraphInterrupt BEFORE
hitl_node executes. The node itself is only executed on resume.

On first astream():
    Graph runs fetch → analyze → [reflect×N] → lint → [refactor] → PAUSE
    astream() stops yielding — no Python exception is raised.
    review_service detects pause via snapshot.next being non-empty.
    Review status set to pending_hitl.

On resume via Command(resume="approved" | "rejected"):
    Graph resumes from MemorySaver checkpoint.
    hitl_node executes — interrupt() returns the injected decision.
    verdict_node produces final verdict and summary.
    GitHub comment posted if approved.

Error Routing
-------------
check_error() routes to hitl_node immediately on upstream failure.
This ensures a human always inspects the result even on partial failure.
The error flag and reason are displayed in verdict_node output.

Routing Controllers
-------------------
check_error()          — after analyze: error → hitl, ok → lint (or reflect)
should_reflect()       — after reflect: loop until REFLECTION_PASSES reached
should_lint_refactor() — after lint: passed → hitl, failed → refactor
should_refactor()      — after validator: passed → hitl, failed → loop (max 2)

Singleton Pattern
-----------------
get_review_graph() returns the cached compiled graph instance.
review_graph module-level alias is required for review_service.py import.
MemorySaver is used for local dev — swap for AsyncPostgresSaver in P6.
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


# ── Quota Control ─────────────────────────────────────────────────────────────

# Number of self-reflection passes after analyze_code_node.
#
# DEVELOPMENT : 0  — skip reflection entirely (1 LLM call per review)
# STAGING     : 1  — one reflection pass      (2 LLM calls per review)
# PRODUCTION  : 2  — full reflection loop     (3 LLM calls per review)
#
# Each pass sends one Gemini request. Free tier = 20 RPD on gemini-2.5-flash-lite.
# At REFLECTION_PASSES=0: 20 full reviews per day.
# At REFLECTION_PASSES=1: 10 full reviews per day.
# At REFLECTION_PASSES=2:  6 full reviews per day.
REFLECTION_PASSES: int = 0


# ── Routing Controllers ───────────────────────────────────────────────────────

def check_error(state: ReviewState) -> str:
    """
    Route after analyze_code_node.

    On error (GitHub fetch failure or hard LLM crash), skip all remaining
    nodes and route directly to hitl_node so a human can inspect the partial
    state before any verdict is issued.

    On success, route to reflect_node if REFLECTION_PASSES > 0, otherwise
    route directly to lint_node to save LLM quota.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "hitl_node"    — upstream error detected
        "reflect_node" — REFLECTION_PASSES > 0 and no error
        "lint_node"    — REFLECTION_PASSES == 0 and no error
    """
    if state.get("error"):
        logger.error(
            "[workflow] check_error: upstream error detected — "
            "reason=%s, skipping reflection+lint, routing to hitl_node",
            state.get("error_reason", "unknown"),
        )
        return "hitl_node"

    if REFLECTION_PASSES > 0:
        logger.debug(
            "[workflow] check_error: no error — "
            "REFLECTION_PASSES=%d, routing to reflect_node",
            REFLECTION_PASSES,
        )
        return "reflect_node"

    logger.info(
        "[workflow] check_error: no error — "
        "REFLECTION_PASSES=0, skipping reflect_node, routing to lint_node"
    )
    return "lint_node"


def should_reflect(state: ReviewState) -> str:
    """
    Control the self-reflection loop after reflect_node.

    Loops until reflection_count reaches REFLECTION_PASSES, then routes
    to lint_node. Each pass sends one LLM request to Gemini.

    Only called when REFLECTION_PASSES > 0. When REFLECTION_PASSES == 0,
    check_error() routes directly to lint_node and this function is never
    invoked.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "reflect_node" — reflection_count < REFLECTION_PASSES
        "lint_node"    — reflection complete
    """
    count = state.get("reflection_count", 0)

    if count < REFLECTION_PASSES:
        logger.info(
            "[workflow] should_reflect: pass %d/%d — routing to reflect_node",
            count + 1,
            REFLECTION_PASSES,
        )
        return "reflect_node"

    logger.info(
        "[workflow] should_reflect: reflection complete "
        "(count=%d REFLECTION_PASSES=%d) — routing to lint_node",
        count,
        REFLECTION_PASSES,
    )
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    """
    Route after lint_node.

    If lint passed or was skipped (no Python files, no sandbox), proceed
    directly to hitl_node — no refactor needed.

    If lint failed, route to refactor_node to generate a corrective patch.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "hitl_node"     — lint passed or skipped, or upstream error
        "refactor_node" — lint failed, refactor attempt needed
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_lint_refactor: error in state — "
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
        "[workflow] should_lint_refactor: lint FAILED — "
        "routing to refactor_node"
    )
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    """
    Control the refactor/validation loop after validator_node.

    Loops until validation passes or refactor_count reaches 2.
    After max attempts, routes to hitl_node regardless — human
    reviews the partial output.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "hitl_node"     — validation passed, max attempts reached, or error
        "refactor_node" — validation failed and attempts remaining
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_refactor: error in state — "
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
            "[workflow] should_refactor: validation FAILED — "
            "retrying refactor (attempt %d/2)",
            count + 1,
        )
        return "refactor_node"

    logger.warning(
        "[workflow] should_refactor: max refactor attempts reached (%d/2) — "
        "routing to hitl_node regardless of validation result",
        count,
    )
    return "hitl_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-In-The-Loop gate node.

    IMPORTANT: This node is NEVER executed on the first graph run.
    interrupt_before=["hitl_node"] in compile() causes astream() to pause
    BEFORE this node executes. The pause is detected by review_service via
    snapshot.next being non-empty — no Python exception is raised.

    This node only executes on resume via Command(resume=decision).
    At that point, interrupt() returns the decision string injected by
    review_service.decide_review() and the node writes it to state.

    Guard Clause
    ------------
    If human_decision is already set (duplicate resume attempt), the node
    returns {} immediately without calling interrupt() again. This prevents
    double-interrupt crashes on retry scenarios.

    Parameters
    ----------
    state : ReviewState
        Graph state at resume time. Contains all accumulated node outputs.

    Returns
    -------
    dict
        {"human_decision": "approved" | "rejected"}  — on normal resume
        {}                                            — if already decided
    """
    pr_number = state.get("pr_number")

    logger.info(
        "[hitl_node] Executing for PR #%s — "
        "graph was resumed via Command(resume=...)",
        pr_number,
    )

    # Guard — already decided on a previous resume attempt
    if state.get("human_decision") is not None:
        logger.info(
            "[hitl_node] human_decision already set to '%s' — "
            "skipping interrupt(), returning empty dict to avoid double-resume",
            state["human_decision"],
        )
        return {}

    logger.info(
        "[hitl_node] Calling interrupt() — "
        "awaiting decision from review_service.decide_review()"
    )

    decision = interrupt(
        "Review paused. Awaiting human decision: approve / reject."
    )

    logger.info(
        "[hitl_node] Decision received — decision='%s' PR #%s",
        decision, pr_number,
    )

    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the P3 LangGraph StateGraph.

    Node Registration
    -----------------
    All eight nodes are registered. reflect_node is registered even when
    REFLECTION_PASSES=0 because LangGraph requires all declared nodes to
    exist. check_error() simply never routes to it when passes=0.

    Conditional Edges
    -----------------
    check_error()     after analyze_code_node — routes to lint or reflect or hitl
    should_reflect()  after reflect_node      — loops or proceeds to lint
    should_lint_refactor() after lint_node    — routes to refactor or hitl
    should_refactor() after validator_node    — loops or proceeds to hitl

    Compilation Options
    -------------------
    checkpointer=MemorySaver()
        REQUIRED for interrupt_before to work. Graph state is persisted
        so it can be resumed after the HITL pause. MemorySaver is in-process
        (lost on restart). Swap for AsyncPostgresSaver in P6 for persistence.

    interrupt_before=["hitl_node"]
        CRITICAL. Without this, interrupt() inside hitl_node has no effect
        and the graph runs straight through to verdict_node without pausing.
        This causes astream() to stop yielding before hitl_node executes.
        Detected by review_service via snapshot.next == ["hitl_node"].

    Returns
    -------
    CompiledStateGraph
        Ready for graph.astream() and graph.ainvoke(Command(resume=...)).

    Raises
    ------
    Exception
        Re-raised on compilation failure — server startup should fail loudly
        rather than silently serving a broken graph.
    """
    try:
        logger.info(
            "[workflow] Building LangGraph workflow — "
            "REFLECTION_PASSES=%d",
            REFLECTION_PASSES,
        )

        builder = StateGraph(ReviewState)

        # ── Register nodes ────────────────────────────────────────────────────
        builder.add_node("fetch_diff_node",   fetch_diff_node)
        builder.add_node("analyze_code_node", analyze_code_node)
        builder.add_node("reflect_node",      reflect_node)    # registered but
        builder.add_node("lint_node",         lint_node)       # may be skipped
        builder.add_node("refactor_node",     refactor_node)   # when PASSES=0
        builder.add_node("validator_node",    validator_node)
        builder.add_node("hitl_node",         hitl_node)
        builder.add_node("verdict_node",      verdict_node)

        # ── Entry point ───────────────────────────────────────────────────────
        builder.set_entry_point("fetch_diff_node")

        # ── Core flow ─────────────────────────────────────────────────────────

        # Step 1: fetch → analyze
        builder.add_edge("fetch_diff_node", "analyze_code_node")

        # Step 2: analyze → (error→hitl | PASSES=0→lint | PASSES>0→reflect)
        builder.add_conditional_edges(
            "analyze_code_node",
            check_error,
            {
                "hitl_node":    "hitl_node",
                "reflect_node": "reflect_node",
                "lint_node":    "lint_node",
            },
        )

        # Step 3: reflect → (loop | lint)  — only reached when PASSES > 0
        builder.add_conditional_edges(
            "reflect_node",
            should_reflect,
            {
                "reflect_node": "reflect_node",
                "lint_node":    "lint_node",
            },
        )

        # Step 4: lint → (hitl | refactor)
        builder.add_conditional_edges(
            "lint_node",
            should_lint_refactor,
            {
                "hitl_node":     "hitl_node",
                "refactor_node": "refactor_node",
            },
        )

        # Step 5: refactor → validate
        builder.add_edge("refactor_node", "validator_node")

        # Step 6: validate → (loop | hitl)
        builder.add_conditional_edges(
            "validator_node",
            should_refactor,
            {
                "refactor_node": "refactor_node",
                "hitl_node":     "hitl_node",
            },
        )

        # Step 7: hitl → verdict → END
        builder.add_edge("hitl_node",    "verdict_node")
        builder.add_edge("verdict_node", END)

        # ── Compile with HITL interrupt ───────────────────────────────────────
        memory = MemorySaver()

        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_node"],
        )

        logger.info(
            "[workflow] LangGraph workflow compiled successfully — "
            "interrupt_before=['hitl_node'] REFLECTION_PASSES=%d",
            REFLECTION_PASSES,
        )
        return graph

    except Exception:
        logger.exception("[workflow] Graph compilation failed")
        raise


# ── Singleton Management ──────────────────────────────────────────────────────

_review_graph = None


def get_review_graph():
    """
    Lazily initialize and return the singleton compiled graph.

    The graph is built once on the first call and cached. Subsequent calls
    return the cached instance. This avoids repeated StateGraph compilation
    and MemorySaver instantiation on every request.

    The singleton also ensures the MemorySaver checkpointer is shared across
    all requests — critical for HITL resume to work, since the checkpoint
    written during trigger_review() must be readable by decide_review().

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


# Module-level alias — required for:
#   from app.graph.workflow import review_graph
# Used by review_service.py trigger_review() and decide_review().
review_graph = get_review_graph()