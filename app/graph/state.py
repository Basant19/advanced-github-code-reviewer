"""
app/graph/state.py

LangGraph Agent State
----------------------
Defines ReviewState — the single shared object that flows through
every node in the LangGraph review workflow.

Every node receives the current state and returns a partial update.
LangGraph merges the updates automatically.

State lifecycle:
    Webhook trigger
        → fetch_diff_node    fills: metadata, diff, files
        → analyze_code_node  fills: issues, suggestions
        → reflect_node       fills: reflection_count (increments)
        → verdict_node       fills: verdict, summary
        → (post to GitHub PR via GitHubClient)

Fields
------
    owner            GitHub repository owner (e.g. "Basant19")
    repo             GitHub repository name  (e.g. "advanced-github-code-reviewer")
    pr_number        Pull request number
    metadata         PR title, author, branches (from get_pr_metadata)
    diff             Unified diff string passed to the LLM
    files            List of changed files with patches
    issues           Problems found by the LLM
    suggestions      Improvement suggestions from the LLM
    reflection_count How many reflection passes have run (max 2)
    verdict          Final decision: "APPROVE" or "REQUEST_CHANGES"
    summary          Full review text posted as a GitHub PR comment
    repo_context     Relevant code retrieved from ChromaDB vector store
"""

import sys
from typing import TypedDict

from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


class ReviewState(TypedDict):
    """
    Shared state object for the LangGraph PR review workflow.

    All fields are optional at initialisation — nodes fill them
    incrementally as the workflow progresses.
    """

    # ── Input — set once when the review is triggered ─────────────────────
    owner:      str     # GitHub repo owner login
    repo:       str     # GitHub repo name
    pr_number:  int     # Pull request number

    # ── Fetched by fetch_diff_node ─────────────────────────────────────────
    metadata:   dict    # {number, title, author, description, base_branch, head_branch, state}
    diff:       str     # unified diff string → sent to LLM
    files:      list    # [{filename, status, changes, patch}, ...]

    # ── Written by analyze_code_node ──────────────────────────────────────
    issues:      list   # ["Missing type hint on line 12", ...]
    suggestions: list   # ["Consider extracting this into a helper", ...]

    # ── Reflection loop control ────────────────────────────────────────────
    reflection_count: int   # incremented by reflect_node; workflow loops while < 2

    # ── Final output ───────────────────────────────────────────────────────
    verdict: str    # "APPROVE" or "REQUEST_CHANGES"
    summary: str    # formatted review text posted to the GitHub PR

    # ── Vector memory ──────────────────────────────────────────────────────
    repo_context: str   # relevant code chunks retrieved from ChromaDB


def build_initial_state(owner: str, repo: str, pr_number: int) -> ReviewState:
    """
    Creates a clean ReviewState with only the trigger fields set.
    All other fields are given safe defaults so nodes never hit KeyError.

    Args:
        owner:     GitHub repository owner login
        repo:      GitHub repository name
        pr_number: Pull request number to review

    Returns:
        ReviewState with defaults ready for the workflow to populate.

    Raises:
        CustomException if any required input field is missing or invalid.
    """
    logger.info(
        f"Building initial ReviewState for {owner}/{repo}#{pr_number}"
    )

    try:
        if not owner or not isinstance(owner, str):
            raise ValueError("owner must be a non-empty string")
        if not repo or not isinstance(repo, str):
            raise ValueError("repo must be a non-empty string")
        if not isinstance(pr_number, int) or pr_number < 1:
            raise ValueError("pr_number must be a positive integer")

        state: ReviewState = {
            # ── trigger inputs ─────────────────────────────────────────
            "owner":            owner.strip(),
            "repo":             repo.strip(),
            "pr_number":        pr_number,
            # ── populated by fetch_diff_node ───────────────────────────
            "metadata":         {},
            "diff":             "",
            "files":            [],
            # ── populated by analyze_code_node ─────────────────────────
            "issues":           [],
            "suggestions":      [],
            # ── reflection loop ────────────────────────────────────────
            "reflection_count": 0,
            # ── populated by verdict_node ──────────────────────────────
            "verdict":          "",
            "summary":          "",
            # ── populated from ChromaDB ────────────────────────────────
            "repo_context":     "",
        }

        logger.info(
            f"Initial ReviewState built successfully — "
            f"owner={owner}, repo={repo}, pr_number={pr_number}"
        )
        return state

    except ValueError as e:
        logger.error(f"Invalid input for ReviewState: {e}")
        raise CustomException(str(e), sys)

    except Exception as e:
        logger.error(f"Unexpected error building ReviewState: {e}")
        raise CustomException(str(e), sys)