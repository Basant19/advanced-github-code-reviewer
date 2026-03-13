"""
E:\advanced-github-code-reviewer\app\graph\workflow.py

LangGraph workflow definition for Advanced GitHub Code Reviewer.

P1: fetch_diff → analyze_code → reflect (×2 via should_reflect) → verdict → END
P2: rewired to add lint → refactor → validator → should_refactor loop (max 3×)
P3: adds interrupt() call before verdict_node — graph suspends and awaits human
    approval via POST /reviews/{id}/approve or POST /reviews/{id}/reject.

Graph structure (P3):
    START
      → fetch_diff_node
      → analyze_code_node
      → reflect_node          ← loops via should_reflect() up to 2×
      → lint_node
      → refactor_node
      → validator_node        ← loops via should_refactor() up to 3×
      → [interrupt()]         ★ P3: graph pauses here, human decision required
      → verdict_node
      → END
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

# ── Conditional edge helpers ─────────────────────────────────────────────────

def should_reflect(state: ReviewState) -> str:
    """
    Controls the self-reflection loop after analyze_code_node.
    Reflects up to 2 times (reflection_count starts at 0 in P1 loop).
    Returns node name to route to next.
    """
    count = state.get("reflection_count", 0)
    if count < 2:
        logger.debug("should_reflect → reflect_node (count=%d)", count)
        return "reflect_node"
    logger.debug("should_reflect → lint_node (count=%d, exiting reflect loop)", count)
    return "lint_node"


def should_refactor(state: ReviewState) -> str:
    """
    Controls the refactor/validate loop after validator_node.
    - validation passed        → proceed to hitl_node (P3 interrupt gate)
    - validation failed + < 3  → loop back to refactor_node
    - reflection_count >= 3    → proceed regardless (max iterations reached)
    """
    validation_result = state.get("validation_result", {})
    reflection_count = state.get("reflection_count", 0)

    if validation_result.get("passed", False):
        logger.debug("should_refactor → hitl_node (validation passed)")
        return "hitl_node"

    if reflection_count < 3:
        logger.debug(
            "should_refactor → refactor_node (validation failed, count=%d)",
            reflection_count,
        )
        return "refactor_node"

    logger.debug(
        "should_refactor → hitl_node (max iterations reached, count=%d)",
        reflection_count,
    )
    return "hitl_node"


# ── P3: HITL interrupt node ──────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-in-the-Loop gate (P3).

    Calls LangGraph interrupt() which immediately suspends graph execution.
    The graph thread is checkpointed at this point.

    Resumption happens when an admin calls:
        POST /reviews/{id}/approve   → human_decision = 'approved'
        POST /reviews/{id}/reject    → human_decision = 'rejected'

    The FastAPI HITL routes resume the graph by invoking it again with the
    same thread_id and injecting human_decision into state.

    Returns an empty dict — state is not modified here; the decision value
    is injected externally on resume via Command(resume=...) or state update.
    """
    logger.info(
        "hitl_node: suspending graph — awaiting human approval for PR #%s",
        state.get("pr_number"),
    )
    # interrupt() raises a special LangGraph signal that checkpoints the graph.
    # Execution resumes from this exact point when the thread is re-invoked.
    interrupt("Awaiting human approval. Call /approve or /reject to continue.")
    return {}


# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the P3 LangGraph StateGraph.

    MemorySaver is required for interrupt() — it checkpoints state so the
    graph can resume from the exact suspended node.

    Returns:
        Compiled LangGraph application with memory checkpointing enabled.
    """
    builder = StateGraph(ReviewState)

    # ── Register all nodes ───────────────────────────────────────────────────
    builder.add_node("fetch_diff_node", fetch_diff_node)
    builder.add_node("analyze_code_node", analyze_code_node)
    builder.add_node("reflect_node", reflect_node)
    builder.add_node("lint_node", lint_node)
    builder.add_node("refactor_node", refactor_node)
    builder.add_node("validator_node", validator_node)
    builder.add_node("hitl_node", hitl_node)       # ★ P3
    builder.add_node("verdict_node", verdict_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("fetch_diff_node")

    # ── Linear edges ────────────────────────────────────────────────────────
    builder.add_edge("fetch_diff_node", "analyze_code_node")

    # ── Reflection loop: analyze → reflect (×2) → lint ──────────────────────
    builder.add_conditional_edges(
        "analyze_code_node",
        should_reflect,
        {"reflect_node": "reflect_node", "lint_node": "lint_node"},
    )
    builder.add_conditional_edges(
        "reflect_node",
        should_reflect,
        {"reflect_node": "reflect_node", "lint_node": "lint_node"},
    )

    # ── Refactor loop: lint → refactor → validator → [loop | hitl] ──────────
    builder.add_edge("lint_node", "refactor_node")
    builder.add_edge("refactor_node", "validator_node")
    builder.add_conditional_edges(
        "validator_node",
        should_refactor,
        {"refactor_node": "refactor_node", "hitl_node": "hitl_node"},
    )

    # ── HITL → verdict → END ─────────────────────────────────────────────────
    builder.add_edge("hitl_node", "verdict_node")
    builder.add_edge("verdict_node", END)

    # ── Compile with MemorySaver (required for interrupt()) ──────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("P3 graph compiled with MemorySaver and interrupt() gate")
    return graph


# ── Module-level graph instance (imported by services) ──────────────────────
review_graph = build_graph()