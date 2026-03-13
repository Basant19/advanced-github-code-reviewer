"""
E:\advanced-github-code-reviewer\app\graph\state.py
ReviewState TypedDict for LangGraph workflow.

P1: Initial fields — owner, repo, pr_number, thread_id, metadata, files,
    diff, issues, suggestions, repo_context, reflection_count, verdict, summary.
P2: No new fields (lint_result, lint_passed, patch, validation_result written
    directly by lint_node / refactor_node / validator_node using existing keys).
P3: Adds human_decision field — 'approved' | 'rejected' | None.
    Set by HITL approval gate after interrupt() resumes.
"""

from typing import Optional
from typing_extensions import TypedDict


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