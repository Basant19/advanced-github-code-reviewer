"""
app/graph/state.py

ReviewState TypedDict — P4 Production Version
----------------------------------------------
Defines the complete state schema shared across all LangGraph nodes.

LangGraph State Design
----------------------
LangGraph merges partial state updates automatically. Every node returns
a dict containing ONLY the keys it modifies — never the full state.
LangGraph merges these partials into the running state at each step.

This file guarantees all keys exist with safe defaults so nodes never
encounter a KeyError when reading state keys they did not write.

P4 Additions
------------
raw_context   : str  — ungraded text retrieved from ChromaDB by retrieve_context_node
context_grade : str  — "yes" | "no" | "skipped" — grade from grade_context_node

P3 Fields (HITL)
----------------
pending_hitl    : bool         — legacy field, kept for backward compat
human_decision  : Optional[str] — "approved" | "rejected" | None

Flow Overview (P4)
------------------
fetch_diff_node
    → retrieve_context_node  ★ P4
    → grade_context_node     ★ P4
    → analyze_code_node
    → reflect_node
    → lint_node
    → refactor_node
    → validator_node
    → memory_write_node      ★ P4
    → [PAUSE: interrupt_before hitl_node]
    → hitl_node
    → verdict_node
    → END
"""

import sys
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from app.core.exceptions import CustomException


# ── Sandbox Result ────────────────────────────────────────────────────────────

class SandboxResult(TypedDict, total=False):
    """
    Standard structure for Docker sandbox execution results.

    Used by lint_node (run_lint) and validator_node (run_tests).
    Must be JSON serializable — persisted to ReviewStep.output_data.

    Fields
    ------
    passed      : bool   — True if ruff/pytest exited with code 0
    output      : str    — stdout from the sandbox container
    errors      : str    — stderr from the sandbox container
    exit_code   : int    — process exit code (0 = success)
    duration_ms : float  — wall-clock time for sandbox execution
    tool        : str    — "ruff" | "pytest" — identifies which tool ran
    """
    passed:      bool
    output:      str
    errors:      str
    exit_code:   int
    duration_ms: float
    tool:        str


# ── Review State ──────────────────────────────────────────────────────────────

class ReviewState(TypedDict, total=False):
    """
    Complete state schema shared across all LangGraph nodes.

    Design Rules
    ------------
    1. total=False — all keys are optional at the TypedDict level.
       build_initial_state() guarantees all keys exist at runtime.
    2. Nodes return PARTIAL updates — only the keys they modify.
    3. Never mutate state directly inside a node — return new values.
    4. All values must be JSON serializable — persisted to PostgreSQL.

    Key Sections
    ------------
    INPUT        : PR coordinates passed in by trigger_review()
    GITHUB DATA  : diff, files, metadata fetched by fetch_diff_node
    RAG (P4)     : ChromaDB retrieval + grading pipeline state
    LLM ANALYSIS : issues and suggestions from analyze_code_node
    LOOP CONTROL : counters for reflection and refactor loops
    SANDBOX      : lint and validation results from Docker sandbox
    ERROR        : error flag and reason for upstream failures
    HITL         : human decision gate state (P3)
    FINAL OUTPUT : verdict and summary produced by verdict_node
    """

    # ── INPUT ─────────────────────────────────────────────────────────────────
    owner:     str   # GitHub username or organization
    repo:      str   # repository name (without owner prefix)
    pr_number: int   # GitHub Pull Request number (positive)
    thread_id: str   # LangGraph MemorySaver thread identifier

    # ── GITHUB DATA ───────────────────────────────────────────────────────────
    metadata: Dict[str, Any]       # PR metadata: title, author, head_branch
    files:    List[Dict[str, Any]] # changed files: filename, status, patch
    diff:     str                  # unified diff string from GitHub API

    # ── RAG — P4 ──────────────────────────────────────────────────────────────
    raw_context:   str
    """
    Ungraded text retrieved from ChromaDB by retrieve_context_node.
    Contains semantically similar code chunks from the indexed repository.
    Empty string if ChromaDB is unavailable or collection is empty.
    Passed to grade_context_node for relevance grading before use.
    """

    context_grade: str
    """
    Relevance grade assigned by grade_context_node (CRAG pattern).
    Values: "yes" | "no" | "skipped"
    "yes"     — raw_context is relevant, injected into analyze_code_node
    "no"      — raw_context not relevant, repo_context set to empty string
    "skipped" — grade_context_node was skipped (no context or LLM failure)
    """

    # ── LLM ANALYSIS ──────────────────────────────────────────────────────────
    issues:       List[str]  # issues found by analyze_code_node (+ reflect)
    suggestions:  List[str]  # improvement suggestions
    repo_context: str        # graded codebase context injected into analyzer

    # ── LOOP CONTROL ──────────────────────────────────────────────────────────
    reflection_count: int  # incremented by reflect_node, max=REFLECTION_PASSES
    refactor_count:   int  # incremented by validator_node on failure, max=2

    # ── SANDBOX ───────────────────────────────────────────────────────────────
    lint_result:      SandboxResult  # result from sandbox.run_lint()
    lint_passed:      bool           # True if lint passed or was skipped

    patch:            str            # corrective unified diff from refactor_node

    validation_result: SandboxResult  # result from sandbox.run_tests()
    validation_passed: bool           # True if tests passed or were skipped

    # ── ERROR HANDLING ────────────────────────────────────────────────────────
    error:        bool  # True if a node set an unrecoverable error
    error_reason: str   # machine-readable reason: "github_fetch_failed" etc.

    # ── HITL (P3) ─────────────────────────────────────────────────────────────
    pending_hitl:   bool           # legacy field — kept for backward compat
    human_decision: Optional[str]  # "approved" | "rejected" | None

    # ── FINAL OUTPUT ──────────────────────────────────────────────────────────
    verdict: str  # "APPROVE" | "REQUEST_CHANGES" | "HUMAN_REJECTED" | "FAILED"
    summary: str  # full GitHub PR comment markdown produced by verdict_node


# ── Initial State Builder ─────────────────────────────────────────────────────

def build_initial_state(
    owner: str,
    repo: str,
    pr_number: int,
    thread_id: str = "",
) -> ReviewState:
    """
    Validate inputs and construct a complete, safe initial ReviewState.

    Why This Matters
    ----------------
    LangGraph merges partial state updates from each node. If a key is
    missing from the initial state and a node reads it before writing it,
    state.get() returns None instead of the expected default type. This
    causes subtle bugs that are hard to trace.

    build_initial_state() guarantees every key exists with a safe default
    so all nodes can safely call state.get("key", default) without risk.

    Validation Strategy
    -------------------
    Fail fast on invalid input types and empty strings.
    Normalize owner and repo by stripping whitespace.
    All validation errors are wrapped in CustomException.

    Parameters
    ----------
    owner : str
        GitHub username or organization name. Must be non-empty string.
    repo : str
        Repository name without owner prefix. Must be non-empty string.
    pr_number : int
        GitHub Pull Request number. Must be a positive integer.

    Returns
    -------
    ReviewState
        Fully initialized state dict with safe defaults for all keys.

    Raises
    ------
    CustomException
        On invalid input type, empty string, or non-positive pr_number.
    """
    try:
        # ── Type validation ───────────────────────────────────────────────────
        if not isinstance(owner, str):
            raise ValueError(
                f"owner must be str, got {type(owner).__name__}"
            )
        if not isinstance(repo, str):
            raise ValueError(
                f"repo must be str, got {type(repo).__name__}"
            )
        if not isinstance(pr_number, int):
            raise ValueError(
                f"pr_number must be int, got {type(pr_number).__name__}"
            )

        # ── Value normalization ───────────────────────────────────────────────
        owner = owner.strip()
        repo = repo.strip()
        thread_id = thread_id.strip()
        if not owner:
            raise ValueError("owner must not be empty string")
        if not repo:
            raise ValueError("repo must not be empty string")
        if pr_number <= 0:
            raise ValueError(
                f"pr_number must be positive integer, got {pr_number}"
            )

        # ── Build initial state ───────────────────────────────────────────────
        state: ReviewState = {

            # INPUT
            "owner":     owner,
            "repo":      repo,
            "pr_number": pr_number,
            "thread_id": thread_id,

            # GITHUB DATA — populated by fetch_diff_node
            "metadata": {},
            "files":    [],
            "diff":     "",

            # RAG (P4) — populated by retrieve_context_node + grade_context_node
            "raw_context":   "",
            "context_grade": "skipped",

            # LLM ANALYSIS — populated by analyze_code_node
            "issues":       [],
            "suggestions":  [],
            "repo_context": "",

            # LOOP CONTROL
            "reflection_count": 0,
            "refactor_count":   0,

            # SANDBOX — populated by lint_node and validator_node
            "lint_result":  {},
            "lint_passed":  True,

            "patch": "",

            "validation_result": {},
            "validation_passed": False,

            # ERROR HANDLING
            "error":        False,
            "error_reason": "",

            # HITL (P3)
            "pending_hitl":   False,
            "human_decision": None,

            # FINAL OUTPUT — populated by verdict_node after HITL resume
            "verdict": "",
            "summary": "",
        }

        return state

    except Exception as e:
        raise CustomException(f"build_initial_state failed: {str(e)}", sys)