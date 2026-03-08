"""
app/graph/workflow.py

LangGraph Review Workflow
--------------------------
Wires all 4 nodes into a compiled StateGraph with:
    - A reflection loop (reflect_node runs up to 2 times)
    - MemorySaver checkpointing (durable execution / resume on crash)
    - A single public entry point: run_review()

Graph structure:
    START
      ↓
    fetch_diff_node
      ↓
    analyze_code_node
      ↓
    reflect_node  ←──────────────┐
      ↓                           │
    should_reflect()              │
      ├── "reflect_node"  ────────┘  (reflection_count < 2)
      └── "verdict_node"             (reflection_count >= 2)
            ↓
          verdict_node
            ↓
           END

Usage:
    from app.graph.workflow import run_review

    result = run_review(
        owner="Basant19",
        repo="advanced-github-code-reviewer",
        pr_number=7,
    )
    print(result["verdict"])   # "APPROVE" or "REQUEST_CHANGES"
    print(result["summary"])   # full review text ready to post to GitHub
"""

import sys
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import ReviewState, build_initial_state
from app.graph.nodes import (
    fetch_diff_node,
    analyze_code_node,
    reflect_node,
    verdict_node,
)
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Conditional edge function ─────────────────────────────────────────────────

def should_reflect(state: ReviewState) -> str:
    """
    Decides whether the agent should run another reflection pass
    or move on to producing the final verdict.

    Called by LangGraph after every reflect_node execution.

    Returns:
        "reflect_node"  — if reflection_count < 2  (loop again)
        "verdict_node"  — if reflection_count >= 2 (move forward)
    """
    if state["reflection_count"] < 2:
        logger.info(
            f"[should_reflect] Reflection count={state['reflection_count']} "
            f"— looping back for another pass"
        )
        return "reflect_node"

    logger.info(
        f"[should_reflect] Reflection count={state['reflection_count']} "
        f"— proceeding to verdict"
    )
    return "verdict_node"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Uses MemorySaver as the checkpointer so execution state is
    persisted after every node — the workflow can resume from
    the last completed node if the process is interrupted.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    logger.info("Building LangGraph review workflow")

    builder = StateGraph(ReviewState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("fetch_diff_node",    fetch_diff_node)
    builder.add_node("analyze_code_node",  analyze_code_node)
    builder.add_node("reflect_node",       reflect_node)
    builder.add_node("verdict_node",       verdict_node)

    # ── Static edges (always follow this path) ────────────────────────────
    builder.add_edge(START,                "fetch_diff_node")
    builder.add_edge("fetch_diff_node",    "analyze_code_node")
    builder.add_edge("analyze_code_node",  "reflect_node")
    builder.add_edge("verdict_node",       END)

    # ── Conditional edge (reflection loop) ───────────────────────────────
    # After reflect_node runs, call should_reflect() to decide next node.
    builder.add_conditional_edges(
        "reflect_node",
        should_reflect,
        {
            "reflect_node": "reflect_node",   # loop back
            "verdict_node": "verdict_node",   # move forward
        },
    )

    # ── Compile with MemorySaver (persistent execution state) ────────────
    memory    = MemorySaver()
    graph     = builder.compile(checkpointer=memory)

    logger.info("LangGraph review workflow compiled successfully")
    return graph


# ── Singleton graph instance ──────────────────────────────────────────────────
# Built once at module import time and reused for every review.

try:
    review_graph = build_graph()
    logger.info("review_graph ready")
except Exception as e:
    logger.error(f"Failed to build review_graph: {e}")
    raise CustomException(str(e), sys)


# ── Public entry point ────────────────────────────────────────────────────────

def run_review(owner: str, repo: str, pr_number: int) -> ReviewState:
    """
    Runs the full PR review workflow for a given pull request.

    This is the single function called by review_service.py.

    Args:
        owner:     GitHub repository owner login
        repo:      GitHub repository name
        pr_number: Pull request number to review

    Returns:
        Final ReviewState with verdict and summary populated.

    Raises:
        CustomException on any unrecoverable workflow error.
    """
    logger.info(f"[run_review] Starting review — {owner}/{repo}#{pr_number}")

    try:
        initial_state = build_initial_state(owner, repo, pr_number)

        # Each run needs a unique thread_id so MemorySaver tracks it separately.
        config = {
            "configurable": {
                "thread_id": f"{owner}-{repo}-{pr_number}"
            }
        }

        final_state = review_graph.invoke(initial_state, config=config)

        logger.info(
            f"[run_review] Completed — "
            f"verdict={final_state.get('verdict')}, "
            f"issues={len(final_state.get('issues', []))}"
        )
        return final_state

    except CustomException:
        raise     # already logged inside the node that raised it

    except Exception as e:
        logger.error(f"[run_review] Unexpected error: {e}")
        raise CustomException(str(e), sys)