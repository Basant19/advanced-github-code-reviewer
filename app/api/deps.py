"""
app/api/deps.py — FastAPI dependency injection.
"""
import sys
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.db.session import AsyncSessionLocal

logger = get_logger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async SQLAlchemy session.

    Commit is called only if the route completes without raising.
    CustomException propagates untouched so route error handlers see
    the original message and status code.
    Only unexpected infrastructure errors are re-wrapped.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except CustomException:
            # Business logic exception — rollback and re-raise unchanged.
            # Do NOT wrap in another CustomException.
            await session.rollback()
            raise
        except Exception as e:
            await session.rollback()
            logger.error("[deps] Database session error: %s", str(e))
            raise CustomException(f"Database session failed: {e}", sys)


def get_graph():
    """
    Dependency that returns the compiled LangGraph graph singleton.
    Fails loudly if called before lifespan startup completes.
    """
    from app.graph.workflow import get_review_graph
    try:
        return get_review_graph()
    except RuntimeError as e:
        logger.critical(
            "[deps] Graph not initialized — lifespan startup incomplete: %s", str(e)
        )
        raise CustomException(str(e), sys)
    except Exception as e:
        logger.exception("[deps] Unexpected error fetching graph")
        raise CustomException(f"Graph dependency failure: {e}", sys)