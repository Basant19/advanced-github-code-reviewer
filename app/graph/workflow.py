"""
app/graph/workflow.py

LangGraph Review Workflow
--------------------------
Wires all nodes into a compiled StateGraph with:
    - A sandbox validation loop (refactor_node → validator_node, up to 3×)
    - MemorySaver checkpointing (durable execution / resume on crash)
    - A single public entry point: run_review()

P2 Graph structure:
    START
      ↓
    fetch_diff_node
      ↓
    analyze_code_node
      ↓
    lint_node
      ↓
    refactor_node  ←─────────────────────────┐
      ↓                                       │
    validator_node                            │
      ↓                                       │
    should_refactor()                         │
      ├── "refactor_node"  ──────────────────┘  (not passed AND count < MAX)
      └── "verdict_node"                         (passed OR count >= MAX)
            ↓
          verdict_node
            ↓
           END

P1 Graph (preserved for reference — reflects_node + should_reflect still
in this file so existing tests continue to pass):
    START → fetch_diff → analyze → reflect(×2) → verdict → END

MAX_REFACTOR_ITERATIONS: 3
    After 3 attempts the workflow exits to verdict_node regardless of
    whether the patch passes. The PR comment will show failed sandbox
    results — the human reviewer sees exactly why it did not pass.

Usage:
    from app.graph.workflow import run_review

    result = run_review(
        owner="Basant19",
        repo="python_tuts",
        pr_number=1,
    )
    print(result["verdict"])   # "APPROVE" or "REQUEST_CHANGES"
    print(result["summary"])   # full review comment ready to post to GitHub
"""

import sys

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import ReviewState, build_initial_state
from app.graph.nodes import (
    fetch_diff_node,
    analyze_code_node,
    reflect_node,       # kept — used by P1 tests
    lint_node,
    refactor_node,
    validator_node,
    verdict_node,
)
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)

# Maximum refactor → validate iterations before forcing verdict
MAX_REFACTOR_ITERATIONS = 3


# ── P1 conditional edge (kept — P1 tests import this) ────────────────────────

def should_reflect(state: ReviewState) -> str:
    """
    P1 conditional edge — kept so existing tests do not break.
    Not used in the P2 graph.

    Returns "reflect_node" if reflection_count < 2, else "verdict_node".
    """
    if state["reflection_count"] < 2:
        logger.info(
            f"[should_reflect] count={state['reflection_count']} — looping"
        )
        return "reflect_node"

    logger.info(
        f"[should_reflect] count={state['reflection_count']} — to verdict"
    )
    return "verdict_node"


# ── P2 conditional edge ───────────────────────────────────────────────────────

def should_refactor(state: ReviewState) -> str:
    """
    Decides whether to loop back to refactor_node or proceed to verdict.

    Called by LangGraph after every validator_node execution.

    Logic:
      - If validation passed → verdict_node (code is clean)
      - If reflection_count >= MAX → verdict_node (max attempts reached)
      - Otherwise → refactor_node (try again)

    Returns:
        "verdict_node"  — proceed to final verdict
        "refactor_node" — loop back and try again
    """
    validation_result = state.get("validation_result")
    reflection_count  = state["reflection_count"]

    # Validation passed — code is clean, move to verdict
    if validation_result and validation_result.passed:
        logger.info(
            f"[should_refactor] Validation PASSED at iteration "
            f"{reflection_count} — proceeding to verdict"
        )
        return "verdict_node"

    # Max iterations reached — exit loop regardless of result
    if reflection_count >= MAX_REFACTOR_ITERATIONS:
        logger.info(
            f"[should_refactor] Max iterations ({MAX_REFACTOR_ITERATIONS}) "
            f"reached — proceeding to verdict with failed sandbox results"
        )
        return "verdict_node"

    # Still failing — loop back for another refactor attempt
    logger.info(
        f"[should_refactor] Validation FAILED — "
        f"iteration={reflection_count}, max={MAX_REFACTOR_ITERATIONS} — "
        f"looping back to refactor_node"
    )
    return "refactor_node"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the P2 LangGraph StateGraph.

    P2 graph: fetch_diff → analyze → lint → refactor → validate(loop) → verdict

    Uses MemorySaver as the checkpointer so execution state survives
    process restarts. Each review uses thread_id as its checkpoint key.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    logger.info("Building P2 LangGraph review workflow")

    builder = StateGraph(ReviewState)

    # ── Register all nodes ────────────────────────────────────────────────
    builder.add_node("fetch_diff_node",   fetch_diff_node)
    builder.add_node("analyze_code_node", analyze_code_node)
    builder.add_node("lint_node",         lint_node)
    builder.add_node("refactor_node",     refactor_node)
    builder.add_node("validator_node",    validator_node)
    builder.add_node("verdict_node",      verdict_node)

    # ── Static edges ──────────────────────────────────────────────────────
    # These always follow in this direction — no branching
    builder.add_edge(START,                "fetch_diff_node")
    builder.add_edge("fetch_diff_node",    "analyze_code_node")
    builder.add_edge("analyze_code_node",  "lint_node")
    builder.add_edge("lint_node",          "refactor_node")
    builder.add_edge("verdict_node",       END)

    # ── Conditional edge: sandbox validation loop ─────────────────────────
    # After validator_node, call should_refactor() to decide next node.
    # Two possible targets: loop back to refactor_node or exit to verdict.
    builder.add_conditional_edges(
        "validator_node",
        should_refactor,
        {
            "refactor_node": "refactor_node",  # loop — try another patch
            "verdict_node":  "verdict_node",   # exit — passed or max reached
        },
    )

    # ── Compile with MemorySaver ──────────────────────────────────────────
    memory = MemorySaver()
    graph  = builder.compile(checkpointer=memory)

    logger.info("P2 LangGraph review workflow compiled successfully")
    return graph


# ── Singleton graph instance ──────────────────────────────────────────────────
# Built once at module import time. Reused for every review.

try:
    review_graph = build_graph()
    logger.info("review_graph ready")
except Exception as e:
    logger.error(f"Failed to build review_graph: {e}")
    raise CustomException(str(e), sys)


# ── Public entry point ────────────────────────────────────────────────────────

def run_review(owner: str, repo: str, pr_number: int) -> ReviewState:
    """
    Runs the full P2 PR review workflow for a given pull request.

    Called by review_service.py. Returns the final ReviewState with
    verdict and summary populated.

    Args:
        owner:     GitHub repository owner login
        repo:      GitHub repository name
        pr_number: Pull request number to review

    Returns:
        Final ReviewState — verdict, summary, lint_result,
        validation_result all populated.

    Raises:
        CustomException on any unrecoverable workflow error.
    """
    logger.info(f"[run_review] Starting — {owner}/{repo}#{pr_number}")

    try:
        initial_state = build_initial_state(owner, repo, pr_number)

        config = {
            "configurable": {
                "thread_id": f"{owner}-{repo}-{pr_number}"
            }
        }

        final_state = review_graph.invoke(initial_state, config=config)

        logger.info(
            f"[run_review] Completed — "
            f"verdict={final_state.get('verdict')} "
            f"issues={len(final_state.get('issues', []))} "
            f"lint_passed={final_state.get('lint_passed')} "
            f"validation_passed="
            f"{final_state.get('validation_result') and final_state['validation_result'].passed}"
        )
        return final_state

    except CustomException:
        raise  # already logged inside the node that raised it

    except Exception as e:
        logger.error(f"[run_review] Unexpected error: {e}")
        raise CustomException(str(e), sys)