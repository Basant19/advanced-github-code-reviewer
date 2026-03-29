"""
app/graph/workflow.py

LangGraph Workflow Definition — P4 Production Version
------------------------------------------------------
[docstring unchanged — same as previous version]
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from app.core.logger import get_logger
from app.core.exceptions import CustomException
from app.graph.state import ReviewState
from app.graph.nodes import (
    fetch_diff_node,
    retrieve_context_node,
    grade_context_node,
    analyze_code_node,
    reflect_node,
    lint_node,
    refactor_node,
    validator_node,
    memory_write_node,
    verdict_node,
    summary_node,
)


logger = get_logger(__name__)


# ── SandboxResult Serializer Registration ─────────────────────────────────────
# Silences: "Deserializing unregistered type SandboxResult from checkpoint"
#
# LangGraph's JsonPlusSerializer uses msgpack for checkpointing. Custom types
# stored in state (SandboxResult from lint_node/validator_node) must be
# registered so the deserializer can reconstruct them on resume.
#
# API: The correct approach for this LangGraph version is to pass
# allowed_msgpack_modules to MemorySaver rather than calling a class method.
# The try/except handles version differences gracefully.
#
# In P6: swap MemorySaver for AsyncPostgresSaver — registration carries over.

_SANDBOX_ALLOWED_MODULES = None

try:
    from app.sandbox.docker_runner import SandboxResult
    _SANDBOX_ALLOWED_MODULES = {"app.sandbox.docker_runner": SandboxResult}
    logger.info(
        "[workflow] SandboxResult loaded for serializer registration"
    )
except Exception as _e:
    logger.warning(
        "[workflow] Could not import SandboxResult for serializer — "
        "checkpoint deserialization warnings may appear. error=%s", str(_e),
    )


# ── Quota Control ─────────────────────────────────────────────────────────────

REFLECTION_PASSES: int = 0


# ── Routing Controllers ───────────────────────────────────────────────────────

def check_rag_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] check_rag_error: upstream error — "
            "reason=%s, routing to hitl_node",
            state.get("error_reason", "unknown"),
        )
        return "hitl_node"

    context_grade = state.get("context_grade", "skipped")
    repo_context = state.get("repo_context", "")

    logger.info(
        "[workflow] check_rag_error: no error — "
        "context_grade=%s repo_context_chars=%d, routing to analyze_code_node",
        context_grade, len(repo_context),
    )
    return "analyze_code_node"


def check_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] check_error: upstream error — "
            "reason=%s, routing to hitl_node",
            state.get("error_reason", "unknown"),
        )
        return "hitl_node"

    if REFLECTION_PASSES > 0:
        logger.debug(
            "[workflow] check_error: REFLECTION_PASSES=%d, routing to reflect_node",
            REFLECTION_PASSES,
        )
        return "reflect_node"

    logger.info(
        "[workflow] check_error: no error — "
        "REFLECTION_PASSES=0, skipping reflect_node, routing to lint_node"
    )
    return "lint_node"


def should_reflect(state: ReviewState) -> str:
    count = state.get("reflection_count", 0)

    if count < REFLECTION_PASSES:
        logger.info(
            "[workflow] should_reflect: pass %d/%d — routing to reflect_node",
            count + 1, REFLECTION_PASSES,
        )
        return "reflect_node"

    logger.info(
        "[workflow] should_reflect: reflection complete "
        "(count=%d REFLECTION_PASSES=%d) — routing to lint_node",
        count, REFLECTION_PASSES,
    )
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] should_lint_refactor: error — routing to memory_write_node"
        )
        return "memory_write_node"

    if state.get("lint_passed", True):
        logger.info(
            "[workflow] should_lint_refactor: lint passed — "
            "routing to memory_write_node"
        )
        return "memory_write_node"

    logger.warning(
        "[workflow] should_lint_refactor: lint FAILED — routing to refactor_node"
    )
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] should_refactor: error — routing to memory_write_node"
        )
        return "memory_write_node"

    count = state.get("refactor_count", 0)
    passed = state.get("validation_passed", False)

    if passed:
        logger.info(
            "[workflow] should_refactor: validation passed — "
            "routing to memory_write_node"
        )
        return "memory_write_node"

    if count < 2:
        logger.warning(
            "[workflow] should_refactor: validation FAILED — "
            "retrying refactor (attempt %d/2)",
            count + 1,
        )
        return "refactor_node"

    logger.warning(
        "[workflow] should_refactor: max refactor attempts reached (%d/2) — "
        "routing to memory_write_node",
        count,
    )
    return "memory_write_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-In-The-Loop gate node.

    Never executed on first graph run — interrupt_before pauses before it.
    Only executes on resume via Command(resume=decision).
    Guard clause prevents double-interrupt on retry.
    """
    pr_number = state.get("pr_number")

    logger.info(
        "[hitl_node] Executing for PR #%s — "
        "graph was resumed via Command(resume=...)",
        pr_number,
    )

    if state.get("human_decision") is not None:
        logger.info(
            "[hitl_node] human_decision already set to '%s' — "
            "returning empty dict to prevent double-resume",
            state["human_decision"],
        )
        return {}

    logger.info(
        "[hitl_node] Calling interrupt() — "
        "awaiting decision from review_service.decide_review()"
    )

    decision = interrupt(
        "Review paused. Awaiting human decision: approve / reject."
    )

    logger.info(
        "[hitl_node] Decision received — decision='%s' PR #%s",
        decision, pr_number,
    )

    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.graph.state import ReviewState
from app.graph.nodes import (
    fetch_diff_node,
    retrieve_context_node,
    grade_context_node,
    analyze_code_node,
    reflect_node,
    lint_node,
    refactor_node,
    validator_node,
    memory_write_node,
    summary_node,
    verdict_node,
)
from app.core.logger import get_logger
from app.core.exceptions import CustomException # Import your custom exception

logger = get_logger(__name__)

# ... (Keep _SANDBOX_ALLOWED_MODULES and routing functions unchanged) ...

def build_graph():
    """
    Construct and compile the P4 LangGraph StateGraph (12 nodes).
    """
    try:
        logger.info(
            "[workflow] Building LangGraph P4 workflow — REF_PASSES=%d",
            REFLECTION_PASSES,
        )

        builder = StateGraph(ReviewState)

        # ── 1. Register Nodes ────────────────────────────────────────────────
        builder.add_node("fetch_diff_node",       fetch_diff_node)
        builder.add_node("retrieve_context_node", retrieve_context_node)
        builder.add_node("grade_context_node",    grade_context_node)
        builder.add_node("analyze_code_node",     analyze_code_node)
        builder.add_node("reflect_node",          reflect_node)
        builder.add_node("lint_node",             lint_node)
        builder.add_node("refactor_node",         refactor_node)
        builder.add_node("validator_node",        validator_node)
        builder.add_node("memory_write_node",     memory_write_node)
        builder.add_node("summary_node",          summary_node) 
        builder.add_node("hitl_node",             hitl_node)
        builder.add_node("verdict_node",          verdict_node)

        # ── 2. Define Flow (Edges) ───────────────────────────────────────────
        builder.set_entry_point("fetch_diff_node")
        builder.add_edge("fetch_diff_node",       "retrieve_context_node")
        builder.add_edge("retrieve_context_node", "grade_context_node")

        builder.add_conditional_edges(
            "grade_context_node",
            check_rag_error,
            {"summary_node": "summary_node", "analyze_code_node": "analyze_code_node"}
        )

        builder.add_conditional_edges(
            "analyze_code_node",
            check_error,
            {"summary_node": "summary_node", "reflect_node": "reflect_node", "lint_node": "lint_node"}
        )

        builder.add_conditional_edges("reflect_node", should_reflect, {"reflect_node": "reflect_node", "lint_node": "lint_node"})
        builder.add_conditional_edges("lint_node", should_lint_refactor, {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"})
        
        builder.add_edge("refactor_node", "validator_node")
        
        builder.add_conditional_edges("validator_node", should_refactor, {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"})

        # Production sequence
        builder.add_edge("memory_write_node", "summary_node")
        builder.add_edge("summary_node",      "hitl_node")
        builder.add_edge("hitl_node",         "verdict_node")
        builder.add_edge("verdict_node",      END)

        # ── 3. Checkpointer & Compilation (The "Reachable" Way) ──────────────
        
        # Initialize memory with custom type registration
        try:
            memory = MemorySaver(allowed_msgpack_modules=_SANDBOX_ALLOWED_MODULES)
            logger.info("[workflow] MemorySaver initialized with SandboxResult registration")
        except TypeError:
            logger.warning("[workflow] MemorySaver version fallback: skipping module registration")
            memory = MemorySaver()

        # Final Compilation
        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["hitl_node"],
        )

        logger.info("[workflow] P4 LangGraph workflow compiled successfully (12 nodes)")
        return graph

    except Exception as e:
        # This captures ANY error in the steps above and formats it using your logic
        custom_error = CustomException(e, sys)
        logger.error(f"[workflow] Graph compilation failed: {custom_error}")
        raise custom_error

# ── Singleton Management ──────────────────────────────────────────────────────

_review_graph = None


def get_review_graph():
    """Lazily initialize and return the singleton compiled graph."""
    global _review_graph

    if _review_graph is None:
        logger.info("[workflow] Initializing P4 graph lazily")
        _review_graph = build_graph()

    return _review_graph


review_graph = get_review_graph()