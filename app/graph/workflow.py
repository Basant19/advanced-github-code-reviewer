"""
app/graph/workflow.py — P4 Production
LangGraph 12-node workflow. AsyncPostgresSaver for checkpoint persistence.
Graph is built lazily inside init_checkpointer() called from on_startup().
"""

import sys

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

from app.core.exceptions import CustomException
from app.core.logger import get_logger
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

logger = get_logger(__name__)

REFLECTION_PASSES: int = 0

# Both populated by init_checkpointer() called from main.py on_startup().
# Never set directly — always go through init_checkpointer() or _use_memory_saver().
_checkpointer = None
_review_graph  = None


# ── Routing Functions ─────────────────────────────────────────────────────────

def check_rag_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] check_rag_error: upstream error=%s — routing to hitl_node",
            state.get("error_reason"),
        )
        return "hitl_node"
    logger.info(
        "[workflow] check_rag_error: ok context_grade=%s — routing to analyze_code_node",
        state.get("context_grade"),
    )
    return "analyze_code_node"


def check_error(state: ReviewState) -> str:
    if state.get("error"):
        logger.error(
            "[workflow] check_error: upstream error=%s — routing to hitl_node",
            state.get("error_reason"),
        )
        return "hitl_node"
    if REFLECTION_PASSES > 0:
        logger.debug("[workflow] check_error: REFLECTION_PASSES=%d — routing to reflect_node", REFLECTION_PASSES)
        return "reflect_node"
    logger.info("[workflow] check_error: REFLECTION_PASSES=0 — routing to lint_node")
    return "lint_node"


def should_reflect(state: ReviewState) -> str:
    count = state.get("reflection_count", 0)
    if count < REFLECTION_PASSES:
        logger.info("[workflow] should_reflect: pass %d/%d — routing to reflect_node", count + 1, REFLECTION_PASSES)
        return "reflect_node"
    logger.info("[workflow] should_reflect: complete count=%d — routing to lint_node", count)
    return "lint_node"


def should_lint_refactor(state: ReviewState) -> str:
    if state.get("error"):
        logger.error("[workflow] should_lint_refactor: error — routing to memory_write_node")
        return "memory_write_node"
    if state.get("lint_passed", True):
        logger.info("[workflow] should_lint_refactor: lint passed — routing to memory_write_node")
        return "memory_write_node"
    logger.warning("[workflow] should_lint_refactor: lint FAILED — routing to refactor_node")
    return "refactor_node"


def should_refactor(state: ReviewState) -> str:
    if state.get("error"):
        logger.error("[workflow] should_refactor: error — routing to memory_write_node")
        return "memory_write_node"
    if state.get("validation_passed", False):
        logger.info("[workflow] should_refactor: validation passed — routing to memory_write_node")
        return "memory_write_node"
    count = state.get("refactor_count", 0)
    if count < 2:
        logger.warning("[workflow] should_refactor: FAILED attempt=%d/2 — routing to refactor_node", count + 1)
        return "refactor_node"
    logger.warning("[workflow] should_refactor: max attempts reached (%d/2) — routing to memory_write_node", count)
    return "memory_write_node"


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_node(state: ReviewState) -> dict:
    """
    Human-in-the-loop gate. Never runs on first pass (interrupt_before fires).
    Only executes on resume via Command(resume=decision).
    """
    pr_number = state.get("pr_number")
    logger.info("[hitl_node] Executing for PR #%s — resumed via Command(resume=...)", pr_number)

    if state.get("human_decision") is not None:
        logger.info(
            "[hitl_node] human_decision='%s' already set — skipping interrupt()",
            state["human_decision"],
        )
        return {}

    logger.info("[hitl_node] Calling interrupt() — graph will pause until decide_review() resumes it")
    decision = interrupt("Review paused. Awaiting human decision: approve / reject.")
    logger.info("[hitl_node] interrupt() returned — decision='%s' PR #%s", decision, pr_number)
    return {"human_decision": decision}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    """
    Compile the 12-node P4 StateGraph with the given checkpointer.
    SandboxResult stored as plain dict via dataclasses.asdict() — no custom serde needed.
    Called once from init_checkpointer() after the pool is open.
    """
    try:
        logger.info(
            "[workflow] Building P4 graph — REFLECTION_PASSES=%d checkpointer=%s",
            REFLECTION_PASSES, type(checkpointer).__name__,
        )

        builder = StateGraph(ReviewState)

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

        builder.set_entry_point("fetch_diff_node")
        builder.add_edge("fetch_diff_node",       "retrieve_context_node")
        builder.add_edge("retrieve_context_node", "grade_context_node")

        builder.add_conditional_edges(
            "grade_context_node", check_rag_error,
            {"hitl_node": "hitl_node", "analyze_code_node": "analyze_code_node"},
        )
        builder.add_conditional_edges(
            "analyze_code_node", check_error,
            {"hitl_node": "hitl_node", "reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "reflect_node", should_reflect,
            {"reflect_node": "reflect_node", "lint_node": "lint_node"},
        )
        builder.add_conditional_edges(
            "lint_node", should_lint_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"},
        )
        builder.add_edge("refactor_node", "validator_node")
        builder.add_conditional_edges(
            "validator_node", should_refactor,
            {"memory_write_node": "memory_write_node", "refactor_node": "refactor_node"},
        )
        builder.add_edge("memory_write_node", "summary_node")
        builder.add_edge("summary_node",      "hitl_node")
        builder.add_edge("hitl_node",         "verdict_node")
        builder.add_edge("verdict_node",      END)

        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["hitl_node"],
        )

        logger.info(
            "[workflow] P4 compiled — 12 nodes interrupt_before=['hitl_node'] checkpointer=%s",
            type(checkpointer).__name__,
        )
        return graph

    except Exception as e:
        logger.exception("[workflow] Graph compilation failed — %s", str(e))
        raise CustomException(str(e), sys)


# ── Checkpointer Initialization ───────────────────────────────────────────────

async def init_checkpointer(connection_string: str) -> None:
    """
    Open AsyncConnectionPool, initialize AsyncPostgresSaver, run setup(), build graph.

    Called exclusively from main.py on_startup() — never at import time.
    This is the only place _checkpointer and _review_graph are set to their
    final production values.

    connection_string: psycopg3 format
        postgresql://user:pass@host:port/dbname
        Must NOT be postgresql+asyncpg:// — that is for SQLAlchemy only.

    Falls back to MemorySaver on any failure so the app can still start
    in development without a full Postgres checkpointer setup.
    """
    global _checkpointer, _review_graph

    if not connection_string:
        logger.warning(
            "[workflow] init_checkpointer — empty connection string, using MemorySaver"
        )
        _use_memory_saver()
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        logger.info(
            "[workflow] init_checkpointer — opening AsyncConnectionPool max_size=10"
        )

        pool = AsyncConnectionPool(
            conninfo=connection_string,
            max_size=10,
            # autocommit=True: required by AsyncPostgresSaver — it manages its own transactions
            # prepare_threshold=0: disables prepared statements — avoids pgbouncer incompatibility
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,  # open=False so we control exactly when the pool connects
        )
        await pool.open()
        logger.info("[workflow] init_checkpointer — connection pool open")

        _checkpointer = AsyncPostgresSaver(pool)

        # setup() creates the langgraph checkpoint tables if they do not exist.
        # Idempotent — safe to call on every startup.
        await _checkpointer.setup()
        logger.info("[workflow] init_checkpointer — AsyncPostgresSaver.setup() complete")

        _review_graph = build_graph(_checkpointer)

        logger.info(
            "[workflow] AsyncPostgresSaver ready — "
            "checkpoints stored in PostgreSQL, survive server restarts"
        )

    except ImportError as e:
        logger.error(
            "[workflow] init_checkpointer — langgraph-checkpoint-postgres not installed. "
            "Run: pip install langgraph-checkpoint-postgres psycopg[binary,pool]. "
            "Falling back to MemorySaver. error=%s", str(e),
        )
        _use_memory_saver()

    except Exception as e:
        logger.exception(
            "[workflow] init_checkpointer — PostgreSQL connection failed. "
            "Check CHECKPOINTER_DB_URL in .env. Falling back to MemorySaver. error=%s", str(e),
        )
        _use_memory_saver()


def _use_memory_saver() -> None:
    """
    Build graph with MemorySaver. Checkpoints are lost on server restart.
    Only acceptable for development or if Postgres is unavailable.
    """
    global _checkpointer, _review_graph
    from langgraph.checkpoint.memory import MemorySaver

    logger.warning(
        "[workflow] _use_memory_saver — checkpoints stored in RAM only. "
        "Approve/reject will fail after server restart. "
        "Set CHECKPOINTER_DB_URL to enable persistence."
    )
    _checkpointer = MemorySaver()
    _review_graph = build_graph(_checkpointer)


# ── Public Accessor ───────────────────────────────────────────────────────────

def get_review_graph():
    """
    Return the compiled graph singleton.

    Raises RuntimeError if called before init_checkpointer() has run.
    This enforces fail-fast behavior — a missing graph means startup
    ordering is broken and silent MemorySaver fallback would cause
    checkpoints to be lost silently.

    In production: always returns AsyncPostgresSaver-backed graph.
    In tests: call _use_memory_saver() directly in test fixtures.
    """
    global _review_graph
    if _review_graph is None:
        raise RuntimeError(
            "[workflow] get_review_graph() called before init_checkpointer() completed. "
            "This means a request was served before startup finished — "
            "this should be impossible with lifespan management. "
            "If you see this in tests, call _use_memory_saver() in your test fixture."
        )
    return _review_graph