"""
app/graph/workflow.py — P4 Production
LangGraph 12-node workflow with AsyncPostgresSaver persistence.
"""

import sys
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

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

# Constants
REFLECTION_PASSES: int = 0
_checkpointer = None
_review_graph = None

# ── Updated Routing Functions ─────────────────────────────────────────────────

def check_rag_error(state: ReviewState) -> str:
    """Route based on RAG success or infrastructure failure."""
    if state.get("critical_infra_failure"):
        logger.error("[workflow] check_rag_error: Infra Failure — routing to hitl_node")
        return "hitl_node"
    return "analyze_code_node"


def check_error(state: ReviewState) -> str:
    """Route after analysis based on error state or reflection configuration."""
    if state.get("critical_infra_failure"):
        return "hitl_node"
    if REFLECTION_PASSES > 0:
        return "reflect_node"
    return "lint_node"


def should_reflect(state: ReviewState) -> str:
    """Evaluate if additional reflection passes are required."""
    count = state.get("reflection_count", 0)
    if count < REFLECTION_PASSES:
        return "reflect_node"
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    """
    Route after linting. Failures in code logic proceed to summary,
    while infrastructure crashes route to HITL.
    """
    if state.get("critical_infra_failure"):
        logger.error("[workflow] should_lint_refactor: Sandbox Crash — routing to hitl_node")
        return "hitl_node"

    # Even if lint_passed is False, we move to memory_write to report the findings
    return "memory_write_node"


def should_refactor(state: ReviewState) -> str:
    """Determine if a refactor loop is needed or if validation passed."""
    if state.get("critical_infra_failure"):
        return "hitl_node"
    if state.get("validation_passed", False):
        return "memory_write_node"
    
    count = state.get("refactor_count", 0)
    if count < 2:
        return "refactor_node"
    return "memory_write_node"


# ── Singleton & Checkpointer Logic ───────────────────────────────────────────

def _use_memory_saver():
    """Fallback for development/test environments."""
    global _checkpointer, _review_graph
    _checkpointer = MemorySaver()
    _review_graph = build_graph(_checkpointer)
    logger.info("[workflow] MemorySaver fallback active")


async def init_checkpointer(connection_string: str) -> None:
    """
    Initialize AsyncPostgresSaver and compile the graph.
    Enforces psycopg3 pooling for production stability.
    """
    global _checkpointer, _review_graph

    if not connection_string:
        logger.warning("[workflow] No connection string provided, using MemorySaver")
        _use_memory_saver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        logger.info("[workflow] Opening AsyncConnectionPool for Checkpointer")
        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await pool.open()
        
        _checkpointer = AsyncPostgresSaver(pool)
        await _checkpointer.setup()
        
        # Build the graph singleton once checkpointer is verified
        _review_graph = build_graph(_checkpointer)
        logger.info("[workflow] AsyncPostgresSaver ready and ReviewGraph compiled")

    except Exception as e:
        logger.exception("[workflow] Postgres Checkpointer failed, falling back to Memory: %s", str(e))
        _use_memory_saver()


def get_review_graph():
    """
    Accessor for the ReviewGraph instance.
    Enforces that init_checkpointer must have run during app startup.
    """
    global _review_graph
    if _review_graph is None:
        raise RuntimeError(
            "[workflow] get_review_graph() called before init_checkpointer(). "
            "Ensure on_startup hook is correctly configured in main.py."
        )
    return _review_graph


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """Human-in-the-loop interruption point."""
    if state.get("critical_infra_failure"):
        return {"human_decision": "rejected", "verdict": "FAILED"}

    if state.get("human_decision") is not None:
        return {}

    decision = interrupt("Review paused. Awaiting human decision: approve / reject.")
    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    """Constructs the StateGraph with all nodes and conditional logic."""
    try:
        builder = StateGraph(ReviewState)

        # Register Nodes
        builder.add_node("fetch_diff_node", fetch_diff_node)
        builder.add_node("retrieve_context_node", retrieve_context_node)
        builder.add_node("grade_context_node", grade_context_node)
        builder.add_node("analyze_code_node", analyze_code_node)
        builder.add_node("reflect_node", reflect_node)
        builder.add_node("lint_node", lint_node)
        builder.add_node("refactor_node", refactor_node)
        builder.add_node("validator_node", validator_node)
        builder.add_node("memory_write_node", memory_write_node)
        builder.add_node("summary_node", summary_node)
        builder.add_node("hitl_node", hitl_node)
        builder.add_node("verdict_node", verdict_node)

        # Set Workflow Flow
        builder.set_entry_point("fetch_diff_node")
        builder.add_edge("fetch_diff_node", "retrieve_context_node")
        builder.add_edge("retrieve_context_node", "grade_context_node")

        builder.add_conditional_edges(
            "grade_context_node", 
            check_rag_error,
            {"hitl_node": "hitl_node", "analyze_code_node": "analyze_code_node"},
        )
        builder.add_conditional_edges(
            "analyze_code_node", 
            check_error,
            {"hitl_node": "hitl_node", "reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "reflect_node", 
            should_reflect,
            {"reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "lint_node", 
            should_lint_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node", "hitl_node": "hitl_node"},
        )
        builder.add_edge("refactor_node", "validator_node")
        builder.add_conditional_edges(
            "validator_node", 
            should_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node", "hitl_node": "hitl_node"},
        )
        builder.add_edge("memory_write_node", "summary_node")
        builder.add_edge("summary_node", "hitl_node")
        builder.add_edge("hitl_node", "verdict_node")
        builder.add_edge("verdict_node", END)

        # Compile with checkpointer and HITL interrupt
        return builder.compile(
            checkpointer=checkpointer, 
            interrupt_before=["hitl_node"]
        )

    except Exception as e:
        logger.exception("[workflow] Graph compilation failed")
        raise CustomException(str(e), sys)