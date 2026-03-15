"""
app/api/deps.py

FastAPI Dependency Injection
------------------------------
Provides reusable dependencies injected into route handlers via Depends().

P3 update:
    - Switched from sync SessionLocal to AsyncSessionLocal (asyncpg driver)
    - get_db() is now an async generator yielding AsyncSession
    - Sync Session and SessionLocal removed — no sync code paths remain

Current dependencies:
    get_db()  — yields an AsyncSession, closes it after the request

Usage in routes:
    from app.api.deps import get_db

    @router.get("/something")
    async def my_route(db: AsyncSession = Depends(get_db)):
        ...
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.core.logger import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async SQLAlchemy session for the duration of a request.
    Guarantees the session is closed even if an exception occurs.

    Injected automatically by FastAPI via Depends(get_db).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()