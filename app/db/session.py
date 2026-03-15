"""
app/db/session.py

SQLAlchemy Database Session Configuration
------------------------------------------
Creates the async engine and session factory for all FastAPI routes.

Driver: asyncpg
    asyncpg is the native async PostgreSQL driver. It works correctly
    with FastAPI, SQLAlchemy async, uvicorn, and Windows without any
    event loop policy hacks.

    DATABASE_URL in .env must use the asyncpg scheme:
        postgresql+asyncpg://user:password@host:port/dbname

Exports:
    async_engine       — AsyncEngine instance (asyncpg driver)
    engine             — alias for async_engine (used by main.py startup)
    AsyncSessionLocal  — async session factory (used by get_db() in deps.py)

Usage (in deps.py):
    async def get_db():
        async with AsyncSessionLocal() as session:
            yield session

P3 update:
    - Removed sync engine and SessionLocal (no sync code paths remain)
    - Removed _make_async_url() helper (DATABASE_URL is now asyncpg directly)
    - Removed Windows ProactorEventLoop workaround (not needed with asyncpg)
"""

import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Async engine ──────────────────────────────────────────────────────────────
# asyncpg connects natively via asyncio — no event loop compatibility issues.
# pool_pre_ping=True sends a lightweight SELECT 1 before each connection is
# handed out, ensuring stale connections are discarded automatically.

logger.info("Initializing async database engine (asyncpg)")

async_engine = create_async_engine(
    settings.DATABASE_URL,  # must be postgresql+asyncpg://...
    pool_pre_ping=True,
    echo=False,
)

logger.info("Async database engine initialized successfully")

# ── Engine alias ──────────────────────────────────────────────────────────────
# main.py imports `engine` and calls `async with engine.begin()` on startup.
# Exposing async_engine under the name `engine` keeps main.py unchanged.

engine = async_engine

# ── Async session factory ─────────────────────────────────────────────────────
# async_sessionmaker is the async equivalent of sessionmaker.
# expire_on_commit=False — prevents SQLAlchemy from expiring ORM objects after
# commit, which would trigger lazy loads on already-closed sessions.

logger.info("Creating async session factory")

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

logger.info("Async session factory created successfully")
logger.info("Database session configuration complete — driver: asyncpg")