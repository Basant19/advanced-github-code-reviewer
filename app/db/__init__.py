"""
Database Initialization Script

This script creates all database tables defined by SQLAlchemy models.

Usage
-----
Run once during project setup:

    uv run python app/db/init_db.py

This will create all tables inside PostgreSQL.
"""

import sys

from app.db.base import Base
from app.db.session import engine

from app.core.logger import get_logger
from app.core.exceptions import CustomException

# Import models to register metadata
from app.db.models import Repository


logger = get_logger(__name__)


def create_tables():
    """
    Create all database tables registered under SQLAlchemy Base.
    """

    try:
        logger.info("Creating database tables")

        Base.metadata.create_all(bind=engine)

        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error("Database table creation failed")
        raise CustomException(e, sys)


if __name__ == "__main__":
    create_tables()