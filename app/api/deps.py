"""
app/api/deps.py

FastAPI Dependency Injection
------------------------------
Provides reusable dependencies injected into route handlers via Depends().

Current dependencies:
    get_db()  — yields a SQLAlchemy session, closes it after the request

Usage in routes:
    from app.api.deps import get_db

    @router.get("/something")
    def my_route(db: Session = Depends(get_db)):
        ...
"""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.core.logger import get_logger

logger = get_logger(__name__)


def get_db() -> Generator[Session, None, None]:
    """
    Yields a SQLAlchemy database session for the duration of a request.
    Guarantees the session is closed even if an exception occurs.

    Injected automatically by FastAPI via Depends(get_db).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()