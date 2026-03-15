"""
app/graph/state.py

ReviewState TypedDict for LangGraph workflow.

P1: Initial fields — owner, repo, pr_number, thread_id, metadata, files,
    diff, issues, suggestions, repo_context, reflection_count, verdict, summary.
P2: No new fields (lint_result, lint_passed, patch, validation_result written
    directly by lint_node / refactor_node / validator_node using existing keys).
P3: Adds human_decision field — 'approved' | 'rejected' | None.
    Set by HITL approval gate after interrupt() resumes.
    Adds build_initial_state() factory — validates inputs and sets safe defaults.
"""

import sys
from typing import Optional
from typing_extensions import TypedDict

from app.core.exceptions import CustomException


class SandboxResult(TypedDict, total=False):
    """Result returned by Docker sandbox execution (lint or test run)."""
    passed: bool
    output: str
    exit_code: int
    duration_ms: float
    error: Optional[str]


class ReviewState(TypedDict, total=False):
    """
    Shared state dict passed between all LangGraph nodes.

    All fields are optional (total=False) so each node can return only the
    keys it writes; LangGraph merges partial dicts automatically.
    """

    # ── Inputs (set once at graph entry) ────────────────────────────────────
    owner: str                   # GitHub repo owner / org
    repo: str                    # GitHub repo name
    pr_number: int               # Pull request number
    thread_id: str               # LangGraph MemorySaver thread identifier

    # ── Fetched from GitHub (fetch_diff_node) ───────────────────────────────
    metadata: dict               # PR title, author, base/head branch, etc.
    files: list                  # List of changed file paths
    diff: str                    # Full unified diff text

    # ── LLM analysis (analyze_code_node) ────────────────────────────────────
    issues: list                 # Detected code issues
    suggestions: list            # Improvement suggestions
    repo_context: str            # ChromaDB retrieved context (may be empty)

    # ── Reflection loop counter (reflect_node / refactor_node) ──────────────
    reflection_count: int        # Incremented each refactor iteration (max 3)

    # ── Docker sandbox — lint (lint_node) ───────────────────────────────────
    lint_result: SandboxResult   # Full SandboxResult from ruff check
    lint_passed: bool            # Convenience bool from lint_result.passed

    # ── Docker sandbox — refactor / validate (refactor_node, validator_node) ─
    patch: str                   # Gemini-generated corrective unified diff
    validation_result: SandboxResult  # Full SandboxResult from ruff + pytest

    # ── P3: Human-in-the-Loop decision ──────────────────────────────────────
    human_decision: Optional[str]
    """
    Set by the HITL approval gate after the graph resumes from interrupt().

    Values:
        'approved'  — admin approved the AI verdict; GitHub comment will post.
        'rejected'  — admin rejected; review saved as HUMAN_REJECTED, no post.
        None        — default; interrupt() has not yet been resolved.
    """

    # ── Verdict (verdict_node) ───────────────────────────────────────────────
    verdict: str                 # 'APPROVE' | 'REQUEST_CHANGES' | 'HUMAN_REJECTED'
    summary: str                 # Full markdown comment text for GitHub


# ── Factory function ─────────────────────────────────────────────────────────

def build_initial_state(owner: str, repo: str, pr_number: int) -> ReviewState:
    """
    Validate trigger inputs and return a ReviewState with safe defaults.

    This is the canonical way to create the initial state dict before
    passing it to review_graph.ainvoke().  It enforces the invariants
    that the rest of the pipeline assumes:
        - owner and repo are non-empty strings
        - pr_number is a positive integer

    Args:
        owner:      GitHub repository owner or organisation name.
        repo:       GitHub repository name.
        pr_number:  Pull request number (must be a positive integer).

    Returns:
        A ReviewState dict with all fields initialised to safe defaults.

    Raises:
        CustomException: if any of the three inputs fail validation.
    """
    # ── Type checks ───────────────────────────────────────────────────────────
    if not isinstance(owner, str):
        raise CustomException(
            f"owner must be a string, got {type(owner).__name__}", sys
        )
    if not isinstance(repo, str):
        raise CustomException(
            f"repo must be a string, got {type(repo).__name__}", sys
        )
    if not isinstance(pr_number, int):
        raise CustomException(
            f"pr_number must be an int, got {type(pr_number).__name__}", sys
        )

    # ── Value checks ─────────────────────────────────────────────────────────
    owner = owner.strip()
    repo  = repo.strip()

    if not owner:
        raise CustomException("owner must not be empty", sys)
    if not repo:
        raise CustomException("repo must not be empty", sys)
    if pr_number <= 0:
        raise CustomException(
            f"pr_number must be a positive integer, got {pr_number}", sys
        )

    # ── Build state with safe defaults ────────────────────────────────────────
    return ReviewState(
        owner            = owner,
        repo             = repo,
        pr_number        = pr_number,
        metadata         = {},
        diff             = "",
        files            = [],
        issues           = [],
        suggestions      = [],
        reflection_count = 0,
        verdict          = "",
        summary          = "",
        repo_context     = "",
        human_decision   = None,
    )