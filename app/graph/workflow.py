"""
workflow.py 
E:\advanced-github-code-reviewer\app\graph\workflow.py

GRAPH ARCHITECTURE:
✔ HITL Enforcement: Uses 'interrupt_before' to freeze state at hitl_node.
✔ Conditional Logic: Routes errors or successful passes directly to the HITL gate.
✔ Quota Management: Optimized reflection and refactor loops to minimize LLM spend.
✔ Singleton Pattern: Ensures memory checkpointer consistency across requests.
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

# ─────────────────────────────────────────────
# ROUTING LOGIC (CONDITIONAL EDGES)
# ─────────────────────────────────────────────

def check_error(state: ReviewState) -> str:
    """Routes to HITL immediately if a critical node fails."""
    if state.get("error"):
        logger.error(f"[ROUTING] Exception detected: {state.get('error_reason')} -> Redirecting to HITL")
        return "hitl_node"
    return "reflect_node"


def should_reflect(state: ReviewState) -> str:
    """Controls the reflection loop. Limits to 1 pass to save tokens/quota."""
    count = state.get("reflection_count", 0)
    if count < 1:
        logger.info(f"[ROUTING] Starting reflection pass {count + 1}")
        return "reflect_node"
    
    logger.info("[ROUTING] Reflection complete -> Proceeding to Lint")
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    """Determines if the code needs an AI refactor based on linting results."""
    if state.get("lint_passed", True):
        logger.info("[ROUTING] Lint passed -> Proceeding to HITL Gate")
        return "hitl_node"

    logger.warning("[ROUTING] Lint failed -> Moving to Refactor node")
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    """Controls the repair loop. Max 2 attempts to fix linting/test errors."""
    count = state.get("refactor_count", 0)
    
    if state.get("validation_passed", False):
        logger.info("[ROUTING] Refactor validated -> Proceeding to HITL Gate")
        return "hitl_node"

    if count < 2:
        logger.info(f"[ROUTING] Retrying refactor. Attempt {count + 1}")
        return "refactor_node"

    logger.warning("[ROUTING] Max refactor attempts reached without validation -> HITL Gate")
    return "hitl_node"

# ─────────────────────────────────────────────
# HITL GATEKEEPER
# ─────────────────────────────────────────────

def hitl_node(state: ReviewState):
    """
    The Human-In-The-Loop Barrier.
    This node uses the LangGraph 'interrupt' function to pause execution.
    The graph state is saved here until an external 'resume' command is issued.
    """
    # Guard clause: If we already have a decision, don't interrupt again.
    if state.get("human_decision") is not None:
        logger.info(f"[HITL] Resume signal detected: {state['human_decision']}")
        return {}

    logger.info("[HITL] Interrupting workflow. Waiting for manual approval/rejection.")
    
    # 🔥 This is the physical pause point
    decision = interrupt("Please approve or reject the AI-generated review.")

    return {"human_decision": decision}

# ─────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────

def build_graph():
    """Compiles the StateGraph with memory checkpointers and interrupt points."""
    logger.info("Building StateGraph with HITL interruption points")

    builder = StateGraph(ReviewState)

    # Register all nodes
    builder.add_node("fetch_diff_node", fetch_diff_node)
    builder.add_node("analyze_code_node", analyze_code_node)
    builder.add_node("reflect_node", reflect_node)
    builder.add_node("lint_node", lint_node)
    builder.add_node("refactor_node", refactor_node)
    builder.add_node("validator_node", validator_node)
    builder.add_node("hitl_node", hitl_node)
    builder.add_node("verdict_node", verdict_node)

    # Initial flow
    builder.set_entry_point("fetch_diff_node")
    builder.add_edge("fetch_diff_node", "analyze_code_node")

    # Error-aware routing after analysis
    builder.add_conditional_edges(
        "analyze_code_node",
        check_error,
        {
            "reflect_node": "reflect_node",
            "hitl_node": "hitl_node",
        },
    )

    # Reflection Loop
    builder.add_conditional_edges(
        "reflect_node",
        should_reflect,
        {
            "reflect_node": "reflect_node",
            "lint_node": "lint_node",
        },
    )

    # Linting & Refactoring Path
    builder.add_conditional_edges(
        "lint_node",
        should_lint_refactor,
        {
            "refactor_node": "refactor_node",
            "hitl_node": "hitl_node",
        },
    )

    builder.add_edge("refactor_node", "validator_node")

    # Validation Loop
    builder.add_conditional_edges(
        "validator_node",
        should_refactor,
        {
            "refactor_node": "refactor_node",
            "hitl_node": "hitl_node",
        },
    )

    # Final Exit Path through Human Gate
    builder.add_edge("hitl_node", "verdict_node")
    builder.add_edge("verdict_node", END)

    # Initialize Persistence (MemorySaver is standard for local; upgrade to Postgres for prod)
    memory = MemorySaver()

    # Compile with CRITICAL interrupt configuration
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["hitl_node"]  # Forces a stop before executing hitl_node
    )

    logger.info("LangGraph workflow compiled and ready")
    return graph

# ─────────────────────────────────────────────
# SINGLETON MANAGEMENT
# ─────────────────────────────────────────────

_review_graph = None

def get_review_graph():
    """Lazily initializes and returns the singleton graph instance."""
    global _review_graph
    if _review_graph is None:
        _review_graph = build_graph()
    return _review_graph

# Export for external usage
review_graph = get_review_graph()