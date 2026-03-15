"""
E:\advanced-github-code-reviewer\app\graph\workflow.py

LangGraph workflow definition for the Advanced GitHub Code Reviewer.

Pipeline Overview
-----------------
START
  → fetch_diff_node
  → analyze_code_node
  → reflect_node (max 2 iterations)
  → lint_node
  → refactor_node
  → validator_node (max 3 iterations)
  → hitl_node (Human-in-the-loop interrupt)
  → verdict_node
  → END

Key Features
------------
1. Automated PR diff analysis using LLM.
2. Self-reflection loop to improve AI reasoning.
3. Lint + refactor + validation loop.
4. Human-in-the-loop approval gate using LangGraph interrupt().
5. Memory checkpointing using MemorySaver.

This design keeps the workflow deterministic and easy to debug.
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


# ─────────────────────────────────────────────────────────────
# Reflection Loop Controller
# ─────────────────────────────────────────────────────────────

def should_reflect(state: ReviewState) -> str:
    """
    Controls the self-reflection loop.

    The AI is allowed to reflect on its analysis up to 2 times
    to improve reasoning quality.

    Flow:
        analyze_code_node → reflect_node → reflect_node → lint_node
    """
    try:
        count = state.get("reflection_count", 0)

        if count < 2:
            logger.debug("Reflection loop triggered (count=%d)", count)
            return "reflect_node"

        logger.debug("Reflection loop completed (count=%d). Moving to lint.", count)
        return "lint_node"

    except Exception as e:
        logger.exception("Error in should_reflect routing: %s", e)
        return "lint_node"


# ─────────────────────────────────────────────────────────────
# Refactor / Validation Loop Controller
# ─────────────────────────────────────────────────────────────

def should_refactor(state: ReviewState) -> str:
    """
    Controls the refactor-validation loop.

    Logic:
        If validation passes → proceed to HITL approval
        If validation fails → retry refactor (max 3 attempts)
        If max attempts reached → proceed anyway

    This prevents infinite loops during automated refactoring.
    """

    try:
        validation_result = state.get("validation_result")
        reflection_count = state.get("reflection_count", 0)

        passed = False

        if validation_result:
            passed = getattr(validation_result, "passed", False)

        if passed:
            logger.info("Validation passed. Moving to HITL approval stage.")
            return "hitl_node"

        if reflection_count < 3:
            logger.warning(
                "Validation failed. Retrying refactor (attempt=%d)",
                reflection_count,
            )
            return "refactor_node"

        logger.warning(
            "Max refactor attempts reached (%d). Proceeding to HITL approval.",
            reflection_count,
        )
        return "hitl_node"

    except Exception as e:
        logger.exception("Error in should_refactor routing: %s", e)
        return "hitl_node"


# ─────────────────────────────────────────────────────────────
# Human-In-The-Loop Interrupt Node
# ─────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-In-The-Loop approval gate.

    This node pauses the graph execution using LangGraph interrupt().

    The graph will remain suspended until an admin approves or rejects
    the review using the API endpoints:

        POST /reviews/{id}/approve
        POST /reviews/{id}/reject

    LangGraph resumes execution from this exact checkpoint.
    """

    try:
        pr_number = state.get("pr_number")

        logger.info(
            "HITL checkpoint reached. Awaiting approval for PR #%s",
            pr_number,
        )

        interrupt(
            "Review paused. Awaiting human approval via /approve or /reject."
        )

        return {}

    except Exception as e:
        logger.exception("HITL interrupt failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the LangGraph workflow.

    MemorySaver is used to checkpoint the state so the workflow
    can safely resume after human approval.

    Returns
    -------
    Compiled LangGraph graph
    """

    try:
        logger.info("Initializing LangGraph workflow")

        builder = StateGraph(ReviewState)

        # Register nodes
        builder.add_node("fetch_diff_node", fetch_diff_node)
        builder.add_node("analyze_code_node", analyze_code_node)
        builder.add_node("reflect_node", reflect_node)
        builder.add_node("lint_node", lint_node)
        builder.add_node("refactor_node", refactor_node)
        builder.add_node("validator_node", validator_node)
        builder.add_node("hitl_node", hitl_node)
        builder.add_node("verdict_node", verdict_node)

        # Entry point
        builder.set_entry_point("fetch_diff_node")

        # Core workflow
        builder.add_edge("fetch_diff_node", "analyze_code_node")

        # Reflection loop
        builder.add_conditional_edges(
            "analyze_code_node",
            should_reflect,
            {
                "reflect_node": "reflect_node",
                "lint_node": "lint_node",
            },
        )

        builder.add_conditional_edges(
            "reflect_node",
            should_reflect,
            {
                "reflect_node": "reflect_node",
                "lint_node": "lint_node",
            },
        )

        # Refactor loop
        builder.add_edge("lint_node", "refactor_node")
        builder.add_edge("refactor_node", "validator_node")

        builder.add_conditional_edges(
            "validator_node",
            should_refactor,
            {
                "refactor_node": "refactor_node",
                "hitl_node": "hitl_node",
            },
        )

        # HITL → Verdict
        builder.add_edge("hitl_node", "verdict_node")
        builder.add_edge("verdict_node", END)

        # Compile graph with checkpointing
        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory)

        logger.info("LangGraph workflow compiled successfully")

        return graph

    except Exception as e:
        logger.exception("Failed to build LangGraph workflow: %s", e)
        raise


# ─────────────────────────────────────────────────────────────
# Global Graph Instance
# ─────────────────────────────────────────────────────────────

review_graph = build_graph()
