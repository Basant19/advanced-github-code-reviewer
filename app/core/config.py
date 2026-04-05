"""
app/core/config.py

Application Configuration
--------------------------
Loads all environment variables from `.env` using pydantic-settings.

The `.env` path is resolved to an absolute path relative to this file
so it works correctly regardless of the working directory.

After loading, all API keys are written to os.environ so that
third-party libraries (LangChain, LangGraph, ChromaDB, LangSmith)
can find them via their standard os.environ lookups — regardless of
how they instantiate internally at runtime.

Usage:
    from app.core.config import settings

    settings.google_api_key
    settings.github_token
    settings.DATABASE_URL
"""

import os
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)

# Resolve .env path relative to this file — works from any working directory
# This file : app/core/config.py
# .env       : project_root/.env
# Path       : this → parent (core/) → parent (app/) → parent (root/)
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

logger.info(f"Loading .env from: {_ENV_PATH}")

if not _ENV_PATH.exists():
    logger.warning(
        f".env file not found at {_ENV_PATH} — "
        "relying on system environment variables"
    )


class Settings(BaseSettings):
    """
    All application configuration loaded from .env / environment.

    Fields marked Field(...) are required — startup fails immediately
    with a clear error if any are missing.
    """

    # ── Application ───────────────────────────────────────────────────────────
    app_name:    str  = "Advanced GitHub Code Reviewer"
    ENVIRONMENT: str  = "development"
    debug:       bool = True

    # ── AI / Model Keys ───────────────────────────────────────────────────────
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    # ── LangSmith ─────────────────────────────────────────────────────────────
    LANGSMITH_TRACING: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str  = Field(
        default="advanced-github-code-reviewer",
        alias="LANGSMITH_PROJECT",
    )
    langsmith_api_key: str = Field(..., alias="LANGSMITH_API_KEY")

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token:          str = Field(..., alias="GITHUB_TOKEN")
    github_webhook_secret: str = Field(..., alias="GITHUB_WEBHOOK_SECRET")

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(..., alias="DATABASE_URL")
    CHECKPOINTER_DB_URL: str = Field(..., alias="CHECKPOINTER_DB_URL")

    # ── Pydantic Settings config ──────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),  # absolute path — safe from any cwd
        case_sensitive=True,
        extra="ignore",
    )


try:
    settings = Settings()

    # ── Validate all required keys loaded ────────────────────────────────────
    _required = {
        "GOOGLE_API_KEY":        settings.google_api_key,
        "GITHUB_TOKEN":          settings.github_token,
        "LANGSMITH_API_KEY":     settings.langsmith_api_key,
        "GITHUB_WEBHOOK_SECRET": settings.github_webhook_secret,
        "DATABASE_URL":          settings.DATABASE_URL,
        "CHECKPOINTER_DB_URL":    settings.CHECKPOINTER_DB_URL,
    }
    _missing = [k for k, v in _required.items() if not v]
    if _missing:
        raise ValueError(f"Missing required config keys: {_missing}")

    # ── Write API keys to os.environ ─────────────────────────────────────────
    # Required so third-party libraries (LangChain, LangGraph, ChromaDB,
    # LangSmith) find keys via os.environ — their standard lookup mechanism.
    # LangGraph re-instantiates models internally at runtime and those new
    # instances bypass constructor api_key args, reading os.environ directly.
    os.environ["GOOGLE_API_KEY"]       = settings.google_api_key
    os.environ["LANGSMITH_API_KEY"]    = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"]    = settings.langsmith_api_key   # LangChain alias
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING).lower()
    os.environ["LANGCHAIN_PROJECT"]    = settings.langsmith_project

    logger.info("Application configuration loaded successfully")
    logger.info(f"Environment            : {settings.ENVIRONMENT}")
    logger.info(f"GOOGLE_API_KEY loaded  : {bool(settings.google_api_key)}")
    logger.info(f"GITHUB_TOKEN loaded    : {bool(settings.github_token)}")
    logger.info(f"LANGSMITH tracing      : {settings.LANGSMITH_TRACING}")
    logger.info(f"DATABASE_URL loaded    : {bool(settings.DATABASE_URL)}")
    logger.info(f"CHECKPOINTER_DB loaded : {bool(settings.CHECKPOINTER_DB_URL)}")

except Exception as e:
    logger.error(f"Failed to load application configuration: {e}")
    raise CustomException(e, sys)
