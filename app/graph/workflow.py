"""
app/graph/workflow.py

LangGraph Workflow Definition — P4 Production Version
------------------------------------------------------
Defines the complete review pipeline as a stateful directed graph.

Graph Flow (P4)
---------------
START
  → fetch_diff_node
  → retrieve_context_node  ★ P4 — query ChromaDB for similar code chunks
  → grade_context_node     ★ P4 — CRAG: grade relevance of retrieved context
  → analyze_code_node          — Gemini analysis (with graded context if relevant)
  → reflect_node               — self-reflection (REFLECTION_PASSES=0 in dev)
  → lint_node                  — ruff in Docker sandbox
  → refactor_node              — Gemini corrective patch (conditional)
  → validator_node             — ruff + pytest in Docker (conditional)
  → memory_write_node      ★ P4 — write findings to ChromaDB long-term memory
  → [PAUSE: interrupt_before hitl_node]
  → hitl_node                  — human decision gate
  → verdict_node               — final verdict + GitHub comment markdown
  → END

P4 RAG Pipeline
---------------
retrieve_context_node queries ChromaDB using the PR diff as the search vector.
If ChromaDB is empty (before POST /repos/index is called), raw_context is "".

grade_context_node (CRAG pattern) asks Gemini whether the retrieved context
is actually relevant to this PR. Grades: "yes" | "no" | "skipped".
If "yes" → repo_context injected into analyze_code_node prompt.
If "no"  → repo_context="" (irrelevant context discarded).
If LLM fails → fails open (context passed through unchanged).

memory_write_node runs AFTER validation, BEFORE hitl_node.
It writes review findings to ChromaDB so future reviews on the same repo
benefit from accumulated review history.

Routing After RAG Nodes
-----------------------
check_rag_error() routes after grade_context_node:
    error → hitl_node (skip remaining nodes)
    ok    → analyze_code_node

check_error() routes after analyze_code_node:
    error → hitl_node
    ok    → reflect_node or lint_node (based on REFLECTION_PASSES)

Reflection Loop
---------------
REFLECTION_PASSES controls how many self-reflection passes run after
analyze_code_node. Set to 0 in development to conserve free-tier quota.

    REFLECTION_PASSES = 0  → 1 LLM call  (analyze only)
    REFLECTION_PASSES = 1  → 2 LLM calls (analyze + reflect)
    REFLECTION_PASSES = 2  → 3 LLM calls (analyze + reflect×2)

P4 LLM Call Budget
------------------
With REFLECTION_PASSES=0 and non-empty ChromaDB:
    grade_context_node  : 1 call  (grades retrieved context)
    analyze_code_node   : 1 call  (main analysis)
    TOTAL               : 2 calls per review

With REFLECTION_PASSES=0 and empty ChromaDB (skips grading):
    analyze_code_node   : 1 call
    TOTAL               : 1 call per review

Free tier: 20 RPD on gemini-2.5-flash-lite.
At 2 calls/review: 10 reviews/day with RAG active.
At 1 call/review:  20 reviews/day without RAG (empty ChromaDB).

HITL Design
-----------
interrupt_before=["hitl_node"] is the ONLY correct mechanism for pausing.
astream() stops yielding when the graph reaches hitl_node.
No Python exception is raised — pause detected via snapshot.next.
review_service.trigger_review() checks snapshot.next after stream ends.

On resume via Command(resume="approved" | "rejected"):
    Graph resumes from MemorySaver checkpoint.
    hitl_node executes — interrupt() returns the injected decision.
    verdict_node produces final verdict and summary.

Singleton Pattern
-----------------
get_review_graph() returns the cached compiled graph.
review_graph module-level alias required for review_service.py import.
MemorySaver used for local dev — swap for AsyncPostgresSaver in P6.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.graph.state import ReviewState
from app.graph.nodes import (
    fetch_diff_node,
    retrieve_context_node,
    grade_context_node,
    analyze_code_node,
    reflect_node,
    lint_node,
    refactor_node,
    validator_node,
    memory_write_node,
    verdict_node,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Quota Control ─────────────────────────────────────────────────────────────

# Number of self-reflection passes after analyze_code_node.
#
# DEVELOPMENT : 0  — skip reflection entirely (1-2 LLM calls per review)
# STAGING     : 1  — one reflection pass      (2-3 LLM calls per review)
# PRODUCTION  : 2  — full reflection loop     (3-4 LLM calls per review)
#
# Free tier = 20 RPD on gemini-2.5-flash-lite.
# At REFLECTION_PASSES=0 with RAG active: ~10 reviews/day.
# At REFLECTION_PASSES=0 without RAG:    ~20 reviews/day.
REFLECTION_PASSES: int = 0


# ── Routing Controllers ───────────────────────────────────────────────────────

def check_rag_error(state: ReviewState) -> str:
    """
    Route after grade_context_node (P4).

    If an upstream error was set by fetch_diff_node or retrieve_context_node,
    skip analyze_code_node and route directly to hitl_node.

    On success, always routes to analyze_code_node regardless of whether
    context was retrieved — analyze_code_node handles empty repo_context
    gracefully by omitting the context section from its prompt.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "hitl_node"        — upstream error detected
        "analyze_code_node" — normal path (with or without context)
    """
    if state.get("error"):
        logger.error(
            "[workflow] check_rag_error: upstream error detected — "
            "reason=%s, routing to hitl_node",
            state.get("error_reason", "unknown"),
        )
        return "hitl_node"

    context_grade = state.get("context_grade", "skipped")
    repo_context = state.get("repo_context", "")

    logger.info(
        "[workflow] check_rag_error: no error — "
        "context_grade=%s repo_context_chars=%d, routing to analyze_code_node",
        context_grade,
        len(repo_context),
    )
    return "analyze_code_node"


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
        "hitl_node"     — upstream error detected
        "reflect_node"  — REFLECTION_PASSES > 0 and no error
        "lint_node"     — REFLECTION_PASSES == 0 and no error
    """
    if state.get("error"):
        logger.error(
            "[workflow] check_error: upstream error detected — "
            "reason=%s, routing to hitl_node",
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

    Only reachable when REFLECTION_PASSES > 0. When REFLECTION_PASSES == 0,
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
    to memory_write_node (P4) before the HITL gate.

    If lint failed, route to refactor_node to generate a corrective patch.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "memory_write_node" — lint passed or skipped, or upstream error
        "refactor_node"     — lint failed, refactor attempt needed
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_lint_refactor: error in state — "
            "routing to memory_write_node then hitl"
        )
        return "memory_write_node"

    if state.get("lint_passed", True):
        logger.info(
            "[workflow] should_lint_refactor: lint passed — "
            "routing to memory_write_node"
        )
        return "memory_write_node"

    logger.warning(
        "[workflow] should_lint_refactor: lint FAILED — "
        "routing to refactor_node"
    )
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    """
    Control the refactor/validation loop after validator_node.

    Loops until validation passes or refactor_count reaches 2.
    After max attempts, routes to memory_write_node (P4) before hitl_node.

    Parameters
    ----------
    state : ReviewState

    Returns
    -------
    str
        "memory_write_node" — validation passed, max attempts reached, or error
        "refactor_node"     — validation failed and attempts remaining
    """
    if state.get("error"):
        logger.error(
            "[workflow] should_refactor: error in state — "
            "routing to memory_write_node"
        )
        return "memory_write_node"

    count = state.get("refactor_count", 0)
    passed = state.get("validation_passed", False)

    if passed:
        logger.info(
            "[workflow] should_refactor: validation passed — "
            "routing to memory_write_node"
        )
        return "memory_write_node"

    if count < 2:
        logger.warning(
            "[workflow] should_refactor: validation FAILED — "
            "retrying refactor (attempt %d/2)",
            count + 1,
        )
        return "refactor_node"

    logger.warning(
        "[workflow] should_refactor: max refactor attempts reached (%d/2) — "
        "routing to memory_write_node",
        count,
    )
    return "memory_write_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-In-The-Loop gate node.

    IMPORTANT: This node is NEVER executed on the first graph run.
    interrupt_before=["hitl_node"] in compile() causes astream() to pause
    BEFORE this node executes. The pause is detected by review_service via
    snapshot.next being non-empty — no Python exception is raised.

    This node only executes on resume via Command(resume=decision).
    interrupt() returns the decision string injected by decide_review()
    and the node writes it to state for verdict_node to read.

    Guard Clause
    ------------
    If human_decision is already set (duplicate resume attempt), returns {}
    immediately to prevent double-interrupt crashes.

    Parameters
    ----------
    state : ReviewState

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

    if state.get("human_decision") is not None:
        logger.info(
            "[hitl_node] human_decision already set to '%s' — "
            "returning empty dict to prevent double-resume",
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
    Construct and compile the P4 LangGraph StateGraph.

    Node Registration (P4 — 11 nodes total)
    ----------------------------------------
    fetch_diff_node       — GitHub API: metadata, files, diff
    retrieve_context_node — ChromaDB query for repo context (P4)
    grade_context_node    — CRAG relevance grader (P4)
    analyze_code_node     — Gemini analysis (context-aware in P4)
    reflect_node          — self-reflection loop (REFLECTION_PASSES=0 in dev)
    lint_node             — ruff in Docker sandbox
    refactor_node         — Gemini corrective patch
    validator_node        — ruff + pytest in Docker
    memory_write_node     — write findings to ChromaDB (P4)
    hitl_node             — human decision gate
    verdict_node          — final verdict + GitHub comment markdown

    Edge Flow
    ---------
    fetch_diff_node
        → retrieve_context_node (always)
        → grade_context_node (always)
        → check_rag_error() → analyze_code_node | hitl_node
        → check_error() → reflect_node | lint_node | hitl_node
        → should_reflect() → reflect_node | lint_node
        → should_lint_refactor() → memory_write_node | refactor_node
        → refactor_node → validator_node
        → should_refactor() → memory_write_node | refactor_node
        → memory_write_node → hitl_node (always)
        → hitl_node → verdict_node (always)
        → verdict_node → END

    Compilation Options
    -------------------
    checkpointer=MemorySaver()
        Required for interrupt_before. State persisted in-process.
        Swap for AsyncPostgresSaver in P6 for cross-restart persistence.

    interrupt_before=["hitl_node"]
        Critical for HITL. Without this, astream() completes without pausing.
        Detected by review_service via snapshot.next == ["hitl_node"].

    Returns
    -------
    CompiledStateGraph
    """
    try:
        logger.info(
            "[workflow] Building LangGraph P4 workflow — "
            "REFLECTION_PASSES=%d",
            REFLECTION_PASSES,
        )

        builder = StateGraph(ReviewState)

        # ── Register all nodes ────────────────────────────────────────────────
        builder.add_node("fetch_diff_node",        fetch_diff_node)
        builder.add_node("retrieve_context_node",  retrieve_context_node)   # P4
        builder.add_node("grade_context_node",     grade_context_node)      # P4
        builder.add_node("analyze_code_node",      analyze_code_node)
        builder.add_node("reflect_node",           reflect_node)
        builder.add_node("lint_node",              lint_node)
        builder.add_node("refactor_node",          refactor_node)
        builder.add_node("validator_node",         validator_node)
        builder.add_node("memory_write_node",      memory_write_node)       # P4
        builder.add_node("hitl_node",              hitl_node)
        builder.add_node("verdict_node",           verdict_node)

        # ── Entry point ───────────────────────────────────────────────────────
        builder.set_entry_point("fetch_diff_node")

        # ── Step 1: fetch → retrieve context ─────────────────────────────────
        builder.add_edge("fetch_diff_node", "retrieve_context_node")

        # ── Step 2: retrieve → grade context ─────────────────────────────────
        builder.add_edge("retrieve_context_node", "grade_context_node")

        # ── Step 3: grade → (error→hitl | ok→analyze) ────────────────────────
        builder.add_conditional_edges(
            "grade_context_node",
            check_rag_error,
            {
                "hitl_node":        "hitl_node",
                "analyze_code_node": "analyze_code_node",
            },
        )

        # ── Step 4: analyze → (error→hitl | PASSES=0→lint | PASSES>0→reflect) ─
        builder.add_conditional_edges(
            "analyze_code_node",
            check_error,
            {
                "hitl_node":    "hitl_node",
                "reflect_node": "reflect_node",
                "lint_node":    "lint_node",
            },
        )

        # ── Step 5: reflect → (loop | lint) ──────────────────────────────────
        builder.add_conditional_edges(
            "reflect_node",
            should_reflect,
            {
                "reflect_node": "reflect_node",
                "lint_node":    "lint_node",
            },
        )

        # ── Step 6: lint → (memory_write | refactor) ─────────────────────────
        builder.add_conditional_edges(
            "lint_node",
            should_lint_refactor,
            {
                "memory_write_node": "memory_write_node",
                "refactor_node":     "refactor_node",
            },
        )

        # ── Step 7: refactor → validate ───────────────────────────────────────
        builder.add_edge("refactor_node", "validator_node")

        # ── Step 8: validate → (loop | memory_write) ─────────────────────────
        builder.add_conditional_edges(
            "validator_node",
            should_refactor,
            {
                "memory_write_node": "memory_write_node",
                "refactor_node":     "refactor_node",
            },
        )

        # ── Step 9: memory_write → hitl (always) ─────────────────────────────
        builder.add_edge("memory_write_node", "hitl_node")

        # ── Step 10: hitl → verdict → END ─────────────────────────────────────
        builder.add_edge("hitl_node",    "verdict_node")
        builder.add_edge("verdict_node", END)

        # ── Compile ───────────────────────────────────────────────────────────
        memory = MemorySaver()

        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_node"],
        )

        logger.info(
            "[workflow] P4 LangGraph workflow compiled — "
            "nodes=11 interrupt_before=['hitl_node'] "
            "REFLECTION_PASSES=%d",
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

    Built once on first call, cached for all subsequent requests.
    The MemorySaver singleton is critical — the checkpoint written by
    trigger_review() must be readable by decide_review() using the
    same MemorySaver instance.

    Returns
    -------
    CompiledStateGraph
    """
    global _review_graph

    if _review_graph is None:
        logger.info("[workflow] Initializing P4 graph lazily")
        _review_graph = build_graph()

    return _review_graph


# Module-level alias — required for:
#   from app.graph.workflow import review_graph
review_graph = get_review_graph()