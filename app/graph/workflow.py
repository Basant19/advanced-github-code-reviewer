"""
app/graph/workflow.py — P4 Production
LangGraph workflow: 12 nodes, HITL gate, SandboxResult stored as dict.
"""

import sys

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.core.exceptions import CustomException
from app.core.logger import get_logger
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
    summary_node,
    verdict_node,
)

logger = get_logger(__name__)

REFLECTION_PASSES: int = 0


# ── Routing ───────────────────────────────────────────────────────────────────

def check_rag_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error("[workflow] check_rag_error: error=%s → hitl_node", state.get("error_reason"))
        return "hitl_node"
    logger.info("[workflow] check_rag_error: ok context_grade=%s → analyze_code_node", state.get("context_grade"))
    return "analyze_code_node"


def check_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error("[workflow] check_error: error=%s → hitl_node", state.get("error_reason"))
        return "hitl_node"
    if REFLECTION_PASSES > 0:
        return "reflect_node"
    logger.info("[workflow] check_error: ok REFLECTION_PASSES=0 → lint_node")
    return "lint_node"


def should_reflect(state: ReviewState) -> str:
    count = state.get("reflection_count", 0)
    if count < REFLECTION_PASSES:
        return "reflect_node"
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    if state.get("error") or state.get("lint_passed", True):
        logger.info("[workflow] should_lint_refactor: lint passed/error → memory_write_node")
        return "memory_write_node"
    logger.warning("[workflow] should_lint_refactor: lint FAILED → refactor_node")
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    if state.get("error") or state.get("validation_passed", False):
        return "memory_write_node"
    count = state.get("refactor_count", 0)
    if count < 2:
        logger.warning("[workflow] should_refactor: FAILED attempt=%d/2 → refactor_node", count + 1)
        return "refactor_node"
    logger.warning("[workflow] should_refactor: max attempts reached → memory_write_node")
    return "memory_write_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """Pause for human decision. Only executes on graph resume."""
    pr_number = state.get("pr_number")
    logger.info("[hitl_node] Executing for PR #%s", pr_number)

    if state.get("human_decision") is not None:
        logger.info("[hitl_node] Already decided — skipping interrupt()")
        return {}

    logger.info("[hitl_node] Calling interrupt() — awaiting human decision")
    decision = interrupt("Review paused. Awaiting human decision: approve / reject.")
    logger.info("[hitl_node] Decision received — decision='%s' PR #%s", decision, pr_number)
    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Build the P4 StateGraph (12 nodes).
    SandboxResult is stored as a plain dict (via dataclasses.asdict in lint_node
    and validator_node) so MemorySaver needs no custom serde registration.
    """
    try:
        logger.info("[workflow] Building LangGraph P4 workflow — REFLECTION_PASSES=%d", REFLECTION_PASSES)

        builder = StateGraph(ReviewState)

        builder.add_node("fetch_diff_node",       fetch_diff_node)
        builder.add_node("retrieve_context_node", retrieve_context_node)
        builder.add_node("grade_context_node",    grade_context_node)
        builder.add_node("analyze_code_node",     analyze_code_node)
        builder.add_node("reflect_node",          reflect_node)
        builder.add_node("lint_node",             lint_node)
        builder.add_node("refactor_node",         refactor_node)
        builder.add_node("validator_node",        validator_node)
        builder.add_node("memory_write_node",     memory_write_node)
        builder.add_node("summary_node",          summary_node)
        builder.add_node("hitl_node",             hitl_node)
        builder.add_node("verdict_node",          verdict_node)

        builder.set_entry_point("fetch_diff_node")
        builder.add_edge("fetch_diff_node",       "retrieve_context_node")
        builder.add_edge("retrieve_context_node", "grade_context_node")

        builder.add_conditional_edges(
            "grade_context_node", check_rag_error,
            {"hitl_node": "hitl_node", "analyze_code_node": "analyze_code_node"},
        )
        builder.add_conditional_edges(
            "analyze_code_node", check_error,
            {"hitl_node": "hitl_node", "reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "reflect_node", should_reflect,
            {"reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "lint_node", should_lint_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"},
        )
        builder.add_edge("refactor_node", "validator_node")
        builder.add_conditional_edges(
            "validator_node", should_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"},
        )
        builder.add_edge("memory_write_node", "summary_node")
        builder.add_edge("summary_node",      "hitl_node")
        builder.add_edge("hitl_node",         "verdict_node")
        builder.add_edge("verdict_node",      END)

        # Plain MemorySaver — no custom serde needed because lint_node and
        # validator_node convert SandboxResult to dict via dataclasses.asdict().
        memory = MemorySaver()

        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_node"],
        )

        logger.info("[workflow] P4 compiled — 12 nodes interrupt_before=['hitl_node']")
        return graph

    except Exception as e:
        logger.exception("[workflow] Graph compilation failed — %s", str(e))
        raise CustomException(str(e), sys)


# ── Singleton ─────────────────────────────────────────────────────────────────

_review_graph = None


def get_review_graph():
    """Return cached compiled graph, building on first call."""
    global _review_graph
    if _review_graph is None:
        logger.info("[workflow] Initializing P4 graph lazily")
        _review_graph = build_graph()
    return _review_graph


review_graph = get_review_graph()