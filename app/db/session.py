"""
Database Session Configuration

This module initializes the SQLAlchemy engine and session factory
used throughout the application.

Responsibilities
---------------
1. Establish connection to PostgreSQL database.
2. Manage database connection pool.
3. Provide session factory for database operations.
4. Integrate logging and structured exception handling.

Architecture
------------
FastAPI
   │
SQLAlchemy Engine
   │
PostgreSQL

Usage
-----
    from app.db.session import SessionLocal

    db = SessionLocal()

    try:
        # perform database operations
        ...
    finally:
        db.close()
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

from app.core.config import settings
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


try:
    logger.info("Initializing database engine")

    engine = create_engine(
        settings.DATABASE_URL,
        echo=True  # prints SQL queries during development
    )

    logger.info("Database engine initialized successfully")

except Exception as e:
    logger.error("Database engine initialization failed")
    raise CustomException(e, sys)


try:
    logger.info("Creating database session factory")

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    logger.info("Session factory created successfully")

except Exception as e:
    logger.error("Session factory creation failed")
    raise CustomException(e, sys)