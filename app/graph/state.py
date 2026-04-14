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
import logging
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from app.core.exceptions import CustomException

# Initialize logger for state tracking
logger = logging.getLogger(__name__)

# ── Sandbox Result ────────────────────────────────────────────────────────────

class SandboxResult(TypedDict, total=False):
    """
    Standard structure for Docker sandbox execution results.
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
    - P4: Added RAG, Grading, and Memory Persistence support.
    """

    # ── INPUT ─────────────────────────────────────────────────────────────────
    owner:     str   
    repo:      str   
    pr_number: int   
    thread_id: str   

    # ── GITHUB DATA ───────────────────────────────────────────────────────────
    metadata:   Dict[str, Any]       
    files:      List[Dict[str, Any]] 
    base_files: Dict[str, str]       # Added in P4 for ground-truth comparison
    diff:       str                  

    # ── RAG — P4 ──────────────────────────────────────────────────────────────
    raw_context:   str
    context_grade: str  # "yes" | "no" | "skipped"

    # ── LLM ANALYSIS ──────────────────────────────────────────────────────────
    issues:       List[str]  
    suggestions:  List[str]  
    repo_context: str        
    original_issues: List[str] # Pre-existing bugs found by Sandbox in base branch

    # ── LOOP CONTROL ──────────────────────────────────────────────────────────
    reflection_count: int  
    refactor_count:   int  

    # ── SANDBOX ───────────────────────────────────────────────────────────────
    lint_result:       SandboxResult  
    lint_passed:       bool           

    patch:             str            

    validation_result: SandboxResult  
    validation_passed: bool           

    # ── ERROR HANDLING ────────────────────────────────────────────────────────
    # "error" is for business/logic errors found in code
    error:                  bool  
    error_reason:           str   
    # "critical_infra_failure" is for system crashes (Docker/API/DB)
    # This prevents discovery of bugs from being treated as system failures.
    critical_infra_failure: bool  

    # ── HITL (P3) ─────────────────────────────────────────────────────────────
    pending_hitl:   bool           
    human_decision: Optional[str]  

    # ── FINAL OUTPUT ──────────────────────────────────────────────────────────
    verdict: str  
    summary: str  


# ── Initial State Builder ─────────────────────────────────────────────────────

def build_initial_state(
    owner: str,
    repo: str,
    pr_number: int,
    thread_id: str = "",
) -> ReviewState:
    """
    Constructs a safe initial ReviewState with production validation.
    """
    try:
        logger.info(
            "[state] Initializing state for %s/%s PR #%d", 
            owner, repo, pr_number
        )

        # ── Type & Value Validation ───────────────────────────────────────────
        if not all(isinstance(x, str) for x in [owner, repo, thread_id]):
            raise ValueError("owner, repo, and thread_id must be strings")
        
        if not isinstance(pr_number, int) or pr_number <= 0:
            raise ValueError(f"Invalid PR number: {pr_number}")

        # Normalization
        owner = owner.strip()
        repo = repo.strip()
        thread_id = thread_id.strip()

        # ── Build Object ──────────────────────────────────────────────────────
        state: ReviewState = {
            "owner":     owner,
            "repo":      repo,
            "pr_number": pr_number,
            "thread_id": thread_id,

            "metadata":   {},
            "files":      [],
            "base_files": {}, 
            "diff":       "",

            "raw_context":   "",
            "context_grade": "skipped",

            "issues":           [],
            "suggestions":      [],
            "repo_context":     "",
            "original_issues":  [], 

            "reflection_count": 0,
            "refactor_count":   0,

            "lint_result":       {},
            "lint_passed":       True,  
            "patch":             "",
            "validation_result": {},
            "validation_passed": False, 

            "error":                  False,
            "error_reason":           "",
            "critical_infra_failure": False, # New safe default

            "pending_hitl":   False,
            "human_decision": None,

            "verdict": "",
            "summary": "",
        }

        return state

    except Exception as e:
        error_msg = f"Failed to initialize ReviewState: {str(e)}"
        logger.error("[state] %s", error_msg)
        raise CustomException(error_msg, sys)