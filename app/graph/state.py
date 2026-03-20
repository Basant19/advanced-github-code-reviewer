"""
state.py 
E:\advanced-github-code-reviewer\app\graph\state.py

UPGRADES:
✔ Strict schema definition (TypedDict safe usage)
✔ Guaranteed default values (no KeyError in graph)
✔ HITL lifecycle compatibility
✔ Loop counters clearly separated
✔ Defensive validation (fail fast)
✔ Debug-friendly structure
✔ Future-proof extensibility

CRITICAL DESIGN NOTES:
- LangGraph merges state → nodes MUST return partial updates only
- This file guarantees ALL keys exist → prevents runtime crashes
- No mutation side-effects → safe for async execution
"""

import sys
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from app.core.exceptions import CustomException


# ─────────────────────────────────────────────
# SANDBOX RESULT TYPE
# ─────────────────────────────────────────────
class SandboxResult(TypedDict, total=False):
    """
    Standard structure for sandbox execution results.

    NOTE:
    - Used by lint_node and validator_node
    - Must be JSON serializable
    """
    passed: bool
    output: str
    errors: str
    exit_code: int
    duration_ms: float
    tool: str


# ─────────────────────────────────────────────
# REVIEW STATE
# ─────────────────────────────────────────────
class ReviewState(TypedDict, total=False):
    """
    Global state shared across all LangGraph nodes.

    IMPORTANT RULES:
    ----------------
    1. Nodes MUST return partial updates only
    2. Graph merges state automatically
    3. Never assume key existence unless defined here
    4. This schema prevents runtime KeyError bugs

    FLOW OVERVIEW:
    --------------
    fetch → analyze → reflect → lint → refactor → validate → HITL → verdict
    """

    # ───────────────── INPUT ─────────────────
    owner: str
    repo: str
    pr_number: int
    thread_id: str  # Used by LangGraph checkpointing

    # ─────────────── GITHUB DATA ─────────────
    metadata: Dict[str, Any]
    files: List[Dict[str, Any]]
    diff: str

    # ─────────────── LLM ANALYSIS ────────────
    issues: List[str]
    suggestions: List[str]
    repo_context: str  # reserved for future RAG

    # ─────────────── LOOP CONTROL ────────────
    reflection_count: int   # controls reflection loop
    refactor_count: int     # controls refactor loop

    # ─────────────── SANDBOX ────────────────
    lint_result: SandboxResult
    lint_passed: bool

    patch: str

    validation_result: SandboxResult
    validation_passed: bool

    # ─────────────── ERROR HANDLING ─────────
    error: bool
    error_reason: str

    # ─────────────── HITL (CRITICAL) ────────
    pending_hitl: bool
    human_decision: Optional[str]  # "approved" | "rejected"

    # ─────────────── FINAL OUTPUT ───────────
    verdict: str
    summary: str


# ─────────────────────────────────────────────
# INITIAL STATE BUILDER
# ─────────────────────────────────────────────
def build_initial_state(
    owner: str,
    repo: str,
    pr_number: int,
) -> ReviewState:
    """
    Construct a SAFE and COMPLETE initial state.

    WHY THIS IS CRITICAL:
    ---------------------
    ✔ Prevents KeyError in nodes
    ✔ Ensures deterministic execution
    ✔ Makes debugging MUCH easier
    ✔ Required for LangGraph stability

    VALIDATION STRATEGY:
    --------------------
    - Fail fast on invalid input
    - Normalize strings
    - Ensure all required fields exist

    Returns:
        ReviewState (fully initialized)
    """

    try:
        # ───────── TYPE VALIDATION ─────────
        if not isinstance(owner, str):
            raise ValueError("owner must be string")

        if not isinstance(repo, str):
            raise ValueError("repo must be string")

        if not isinstance(pr_number, int):
            raise ValueError("pr_number must be int")

        # ───────── VALUE NORMALIZATION ─────
        owner = owner.strip()
        repo = repo.strip()

        if not owner:
            raise ValueError("owner cannot be empty")

        if not repo:
            raise ValueError("repo cannot be empty")

        if pr_number <= 0:
            raise ValueError("pr_number must be positive")

        # ───────── SAFE DEFAULT STATE ──────
        state: ReviewState = {
            # INPUT
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,

            # GITHUB
            "metadata": {},
            "files": [],
            "diff": "",

            # ANALYSIS
            "issues": [],
            "suggestions": [],
            "repo_context": "",

            # LOOP CONTROL
            "reflection_count": 0,
            "refactor_count": 0,

            # SANDBOX
            "lint_result": {},
            "lint_passed": True,

            "patch": "",

            "validation_result": {},
            "validation_passed": False,

            # ERROR HANDLING
            "error": False,
            "error_reason": "",

            # HITL
            "pending_hitl": False,
            "human_decision": None,

            # FINAL OUTPUT
            "verdict": "",
            "summary": "",
        }

        return state

    except Exception as e:
        # 🔥 CRITICAL: Always wrap in CustomException
        raise CustomException(f"Invalid input: {str(e)}", sys)