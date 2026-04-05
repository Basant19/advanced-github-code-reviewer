"""
app/main.py — P4 Production
FastAPI application. PostgreSQL tables + AsyncPostgresSaver initialized on startup.

Run via: python run.py  (NOT uvicorn app.main:app directly)
The run.py entry point sets WindowsSelectorEventLoopPolicy before uvicorn starts.
"""

import time
import traceback

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.base import Base
from app.db.session import engine
from app.api.deps import get_db

# Routers imported at module level — safe because run.py sets the event
# loop policy before uvicorn imports this file.
from app.api.routes import webhook, review, chat
from app.api.routes.hitl import router as hitl_router
from app.api.routes.repos import router as repos_router
from app.services.review_service import list_all_reviews

logger = get_logger(__name__)

app = FastAPI(
    title="Advanced GitHub Code Reviewer",
    description=(
        "Agentic AI platform: LangGraph · Gemini 2.5 Flash Lite · "
        "PostgreSQL · Docker sandbox · ChromaDB RAG.\n\n"
        "**HITL Flow:**\n"
        "1. `POST /reviews/trigger` — AI review, pauses at HITL gate\n"
        "2. `POST /reviews/id/{id}/decision` — approve or reject\n\n"
        "**RAG Flow (P4):**\n"
        "1. `POST /repos/index` — index repository into ChromaDB\n"
        "2. Trigger reviews — context injected automatically"
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Request Logging ───────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(
        "[REQUEST] %s %s — client=%s",
        request.method, request.url.path,
        request.client.host if request.client else "unknown",
    )
    try:
        response = await call_next(request)
        logger.info(
            "[RESPONSE] %s %s → %d (%sms)",
            request.method, request.url.path,
            response.status_code,
            round((time.time() - start) * 1000, 2),
        )
        return response
    except Exception as e:
        logger.exception("[CRASH] %s %s — %s", request.method, request.url.path, str(e))
        raise


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """
    Two-phase startup:

    Phase 1 — SQLAlchemy (asyncpg): create ORM tables. Fatal on failure.
    Phase 2 — AsyncPostgresSaver (psycopg3): create checkpoint tables,
              open pool, build graph. Falls back to MemorySaver on failure.

    This runs AFTER the event loop is created. By the time this executes,
    run.py has already set WindowsSelectorEventLoopPolicy, so psycopg3
    can create async connections successfully.
    """
    logger.info("[startup] Starting Advanced GitHub Code Reviewer P4 — version=4.0.0")

    # ── Phase 1: SQLAlchemy ORM tables ───────────────────────────────────────
    try:
        logger.info("[startup] Phase 1 — initializing SQLAlchemy ORM tables")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("[startup] Phase 1 complete — database schema ready")

    except Exception as e:
        logger.exception("[startup] Phase 1 FAILED — %s", str(e))
        raise RuntimeError(f"Database initialization failed: {e}") from e

    # ── Phase 2: LangGraph AsyncPostgresSaver ─────────────────────────────────
    try:
        from app.graph.workflow import init_checkpointer

        checkpointer_url = settings.CHECKPOINTER_DB_URL or ""

        if not checkpointer_url:
            logger.warning(
                "[startup] CHECKPOINTER_DB_URL not set — MemorySaver fallback. "
                "Checkpoints will not survive restarts."
            )
            await init_checkpointer("")
        else:
            display = checkpointer_url.split("@")[-1] if "@" in checkpointer_url else checkpointer_url
            logger.info("[startup] Phase 2 — initializing AsyncPostgresSaver — %s", display)
            await init_checkpointer(checkpointer_url)

    except Exception as e:
        logger.exception("[startup] Phase 2 unexpected failure — %s", str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        "[startup] Configuration — environment=%s db=%s checkpointer_db=%s langsmith=%s",
        settings.ENVIRONMENT,
        bool(settings.DATABASE_URL),
        bool(settings.CHECKPOINTER_DB_URL),
        settings.LANGSMITH_TRACING,
    )
    logger.info("[startup] Application startup complete")


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException) -> JSONResponse:
    logger.error("[exception_handler] CustomException on %s — %s", request.url.path, str(exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "[exception_handler] Unhandled %s on %s — %s",
        type(exc).__name__, request.url.path, str(exc),
    )
    logger.debug("[exception_handler] Traceback:\n%s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(webhook.router)
app.include_router(review.router)
app.include_router(chat.router)
app.include_router(hitl_router)
app.include_router(repos_router)

logger.info(
    "[startup] Routers registered — /webhook, /reviews, /chat, /reviews (HITL), /repos (P4 RAG)"
)


# ── Dashboard Endpoint ────────────────────────────────────────────────────────

@app.get("/reviews/", tags=["reviews"], summary="List all reviews (dashboard)")
async def get_all_reviews(db: AsyncSession = Depends(get_db)):
    logger.info("[get_all_reviews] Dashboard request")
    try:
        reviews = await list_all_reviews(db)
        logger.info("[get_all_reviews] Returning %d review(s)", len(reviews))
        return reviews
    except CustomException as e:
        logger.error("[get_all_reviews] Service error — %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("[get_all_reviews] Unexpected error — %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"], summary="Health check")
async def health_check() -> dict:
    return {
        "status":      "ok",
        "version":     "4.0.0",
        "phase":       "P4",
        "environment": settings.ENVIRONMENT,
    }