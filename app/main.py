"""
app/main.py — P4 Production
FastAPI app with lifespan-managed startup.
Run via: python run.py
"""

import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.base import Base
from app.db.session import engine
from app.api.deps import get_db

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle manager.

    Startup order is strict — no requests are served until both phases complete.
    Phase 1: SQLAlchemy ORM tables (asyncpg driver). Fatal on failure.
    Phase 2: AsyncPostgresSaver (psycopg3 driver). Fatal on failure.

    Using lifespan instead of @app.on_event("startup") guarantees that
    init_checkpointer() runs BEFORE the first request — there is no window
    where get_review_graph() can return a MemorySaver-backed graph at
    request time.
    """
    logger.info("[startup] Starting Advanced GitHub Code Reviewer P4 — version=4.0.0")

    # ── Phase 1: SQLAlchemy ORM tables ───────────────────────────────────────
    try:
        logger.info("[startup] Phase 1 — initializing SQLAlchemy ORM tables")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("[startup] Phase 1 complete — database schema ready")
    except Exception as e:
        logger.exception("[startup] Phase 1 FAILED — cannot connect to database: %s", str(e))
        raise RuntimeError(f"Database initialization failed: {e}") from e

    # ── Phase 2: LangGraph AsyncPostgresSaver ─────────────────────────────────
    # Imported here to avoid triggering module-level graph construction
    # before the event loop policy is active.
    try:
        from app.graph.workflow import init_checkpointer

        checkpointer_url = settings.CHECKPOINTER_DB_URL or ""
        if not checkpointer_url:
            raise RuntimeError(
                "CHECKPOINTER_DB_URL is not set in .env. "
                "AsyncPostgresSaver requires a psycopg3 connection string. "
                "Example: postgresql://user:pass@localhost:5432/dbname"
            )

        display = checkpointer_url.split("@")[-1] if "@" in checkpointer_url else checkpointer_url
        logger.info("[startup] Phase 2 — initializing AsyncPostgresSaver — %s", display)
        await init_checkpointer(checkpointer_url)

    except RuntimeError:
        raise  # propagate config errors immediately
    except Exception as e:
        logger.exception("[startup] Phase 2 FAILED — %s", str(e))
        raise RuntimeError(f"AsyncPostgresSaver initialization failed: {e}") from e

    # ── Ready ─────────────────────────────────────────────────────────────────
    logger.info(
        "[startup] Configuration — environment=%s db=%s checkpointer_db=%s langsmith=%s",
        settings.ENVIRONMENT,
        bool(settings.DATABASE_URL),
        bool(settings.CHECKPOINTER_DB_URL),
        settings.LANGSMITH_TRACING,
    )
    logger.info("[startup] Application startup complete — serving requests")

    yield  # server is live here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("[shutdown] Application shutting down")


# ── Application ───────────────────────────────────────────────────────────────

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
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Request Logging ───────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info("[REQUEST] %s %s — client=%s",
                request.method, request.url.path,
                request.client.host if request.client else "unknown")
    try:
        response = await call_next(request)
        logger.info("[RESPONSE] %s %s → %d (%sms)",
                    request.method, request.url.path,
                    response.status_code,
                    round((time.time() - start) * 1000, 2))
        return response
    except Exception as e:
        logger.exception("[CRASH] %s %s — %s", request.method, request.url.path, str(e))
        raise


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException) -> JSONResponse:
    logger.error(
        "[exception_handler] CustomException on %s — %s",
        request.url.path, str(exc),
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # RuntimeError "Caught handled exception, but response already started"
    # means a middleware or streaming response already sent headers.
    # Log it but don't try to send another response — that would crash.
    error_msg = str(exc)
    if "response already started" in error_msg.lower():
        logger.warning(
            "[exception_handler] Response already started on %s — "
            "cannot send error response. Original error: %s",
            request.url.path, error_msg,
        )
        # Cannot send a new response — just return to avoid double-send crash
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )

    logger.error(
        "[exception_handler] Unhandled %s on %s — %s",
        type(exc).__name__, request.url.path, str(exc),
    )
    logger.debug("[exception_handler] Traceback:\n%s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})



# ── Routers ───────────────────────────────────────────────────────────────────

from app.api.routes import webhook, review, chat          # noqa: E402
from app.api.routes.hitl import router as hitl_router     # noqa: E402
from app.api.routes.repos import router as repos_router   # noqa: E402
from app.services.review_service import list_all_reviews  # noqa: E402

app.include_router(webhook.router)
app.include_router(review.router)
app.include_router(chat.router)
app.include_router(hitl_router)
app.include_router(repos_router)

logger.info("[startup] Routers registered — /webhook, /reviews, /chat, /reviews (HITL), /repos (P4 RAG)")


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