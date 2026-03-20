"""
app/main.py

FastAPI Application Entry Point (P3 - Production Safe)
-----------------------------------------------------
✔ Structured logging
✔ Robust startup lifecycle
✔ Global exception safety
✔ Clean router registration
✔ Debuggable in production

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import time
import traceback

from app.api.routes import webhook, review, chat
from app.api.routes.hitl import router as hitl_router
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.base import Base
from app.db.session import engine
from app.api.deps import get_db
from app.services.review_service import list_all_reviews


logger = get_logger(__name__)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced GitHub Code Reviewer",
    description=(
        "Agentic AI platform that automatically reviews GitHub Pull Requests "
        "using LangGraph, Gemini, ChromaDB, and PostgreSQL."
    ),
    version="3.0.1",   #  bumped version
)


# ── Middleware (Request Logging) ──────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(
        "[REQUEST] %s %s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        duration = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "[RESPONSE] %s %s -> %s (%sms)",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        return response

    except Exception as e:
        duration = round((time.time() - start_time) * 1000, 2)

        logger.exception(
            "[CRASH] %s %s failed in %sms",
            request.method,
            request.url.path,
            duration,
        )
        raise


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when the server starts.
    Initializes DB and validates critical dependencies.
    """
    logger.info("Starting Advanced GitHub Code Reviewer (P3)")

    try:
        logger.info("Initializing database...")

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info(" Database tables ready")

        #  Optional sanity checks
        logger.info(
            "Environment=%s | DB=%s | LangSmith=%s",
            settings.ENVIRONMENT,
            bool(settings.DATABASE_URL),
            settings.LANGSMITH_TRACING,
        )

    except Exception as e:
        logger.exception(" Startup failed")
        raise RuntimeError("Application startup failed") from e


# ── Global Exception Handlers ─────────────────────────────────────────────────

@app.exception_handler(CustomException)
async def custom_exception_handler(
    request: Request, exc: CustomException
) -> JSONResponse:
    logger.error(
        "[CustomException] path=%s message=%s",
        request.url.path,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.error(
        "[UnhandledException] path=%s error=%s",
        request.url.path,
        str(exc),
    )
    logger.debug(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(webhook.router)
app.include_router(review.router)
app.include_router(chat.router)
app.include_router(hitl_router)

logger.info("Routers registered: /webhook, /reviews, /chat, /reviews (HITL)")


# ── Dashboard endpoint ────────────────────────────────────────────────────────

@app.get("/reviews/", tags=["reviews"], summary="List all reviews (dashboard)")
async def get_all_reviews(db: AsyncSession = Depends(get_db)):
    """
    Returns all reviews across all repositories.
    Used by Streamlit dashboard.
    """
    try:
        logger.info("[GET_ALL_REVIEWS] Fetching all reviews")
        reviews = await list_all_reviews(db)

        logger.info("[GET_ALL_REVIEWS] Returned %d reviews", len(reviews))

        return reviews

    except CustomException as e:
        logger.error("[GET_ALL_REVIEWS] Failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("[GET_ALL_REVIEWS] Unexpected error")
        raise HTTPException(status_code=500, detail="Internal error")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": "3.0.1",
        "env": settings.ENVIRONMENT,
    }