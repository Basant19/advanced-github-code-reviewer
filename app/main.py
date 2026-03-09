"""
app/main.py

FastAPI Application Entry Point
---------------------------------
Creates the FastAPI app, registers all routers, and sets up the database
on startup.

Run the server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API docs (auto-generated):
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc

Registered routers:
    /webhook  ← GitHub webhook events
    /reviews  ← trigger + fetch reviews
    /chat     ← thread-based messaging
"""

import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import webhook, review, chat
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.base import Base
from app.db.session import engine

logger = get_logger(__name__)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced GitHub Code Reviewer",
    description=(
        "Agentic AI platform that automatically reviews GitHub Pull Requests "
        "using LangGraph, Gemini, ChromaDB, and PostgreSQL."
    ),
    version="1.0.0",
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup() -> None:
    """
    Runs once when the server starts.
    Creates all database tables if they don't exist yet.
    """
    logger.info("Starting Advanced GitHub Code Reviewer")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(CustomException)
async def custom_exception_handler(
    request: Request, exc: CustomException
) -> JSONResponse:
    """Converts unhandled CustomExceptions to 500 JSON responses."""
    logger.error(f"Unhandled CustomException on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(webhook.router)
app.include_router(review.router)
app.include_router(chat.router)

logger.info("Routers registered: /webhook, /reviews, /chat")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Returns server status. Used by load balancers and uptime monitors."""
    return {"status": "ok", "version": "1.0.0"}