"""
app/api/deps.py - FastAPI Dependency Injection (P5 Updated)
"""

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.core.logger import get_logger
from app.graph.workflow import get_review_graph  # Your graph builder function

logger = get_logger(__name__)

# --- SINGLETON GRAPH INSTANCE ---
# We compile the graph once when the app starts.
# This ensures all threads/requests share the same state logic.
_compiled_graph = get_review_graph()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async SQLAlchemy session.
    Guarantees the session is closed after the request.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            # Note: 'async with' handles closing, but explicit close is safe
            await session.close()

def get_graph():
    """
    Dependency that returns the pre-compiled LangGraph instance.
    Injected via Depends(get_graph).
    """
    if _compiled_graph is None:
        logger.error("LangGraph was not initialized correctly!")
        # Fallback to compilation if needed, though get_workflow should handle it
        return get_review_graph()
    return _compiled_graph