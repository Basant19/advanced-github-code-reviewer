"""
app/main.py

FastAPI Application Entry Point (P3)
--------------------------------------
Creates the FastAPI app, registers all routers, and sets up the database
on startup.

Run the server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API docs (auto-generated):
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc

Registered routers:
    /webhook      ← GitHub webhook events
    /reviews      ← trigger + fetch reviews + HITL approve/reject  ★ P3
    /chat         ← thread-based messaging

P3 changes:
    - hitl_router registered under /reviews prefix
    - GET /reviews/ added for Streamlit dashboard listing
    - on_startup converted to async (DB uses async engine in P3)
"""

from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import webhook, review, chat
from app.api.routes.hitl import router as hitl_router       # ★ P3
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.base import Base
from app.db.session import engine
from app.api.deps import get_db                             # ← correct location
from app.services.review_service import list_all_reviews    # ★ P3


    
logger = get_logger(__name__)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced GitHub Code Reviewer",
    description=(
        "Agentic AI platform that automatically reviews GitHub Pull Requests "
        "using LangGraph, Gemini, ChromaDB, and PostgreSQL."
    ),
    version="3.0.0",
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when the server starts.
    Creates all database tables if they don't exist yet.
    """
    logger.info("Starting Advanced GitHub Code Reviewer (P3)")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ready")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        raise


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(CustomException)
async def custom_exception_handler(
    request: Request, exc: CustomException
) -> JSONResponse:
    """Converts unhandled CustomExceptions to 500 JSON responses."""
    logger.error("Unhandled CustomException on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(webhook.router)
app.include_router(review.router)
app.include_router(chat.router)
app.include_router(hitl_router)    # ★ P3 — adds /reviews/{id}/approve|reject|status

logger.info("Routers registered: /webhook, /reviews, /chat, /reviews (HITL)")


# ── Dashboard endpoint — GET /reviews/ ───────────────────────────────────────
# Used by the Streamlit dashboard to list all reviews across all repositories.
# Returns a flat list of dicts enriched with pr_number, repo_name, pr_title.

@app.get("/reviews/", tags=["reviews"], summary="List all reviews (dashboard)")
async def get_all_reviews(db: AsyncSession = Depends(get_db)):
    """
    Returns all reviews across all repositories, newest first.
    Used by the Streamlit admin dashboard for the PR list view.
    """
    try:
        return await list_all_reviews(db)
    except CustomException as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Returns server status. Used by load balancers and uptime monitors."""
    return {"status": "ok", "version": "3.0.0"}