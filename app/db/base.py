"""
SQLAlchemy Declarative Base

This module defines the base class used by all ORM models
in the Advanced GitHub Code Reviewer application.

Purpose
-------
SQLAlchemy requires a common base class that every model
inherits from. This base class maintains a registry of
all database tables and metadata.

Why This Is Important
---------------------
1. Centralizes metadata for all database tables.
2. Allows SQLAlchemy to automatically generate schema.
3. Enables migration tools like Alembic.
4. Keeps database architecture modular.

Example
-------
    from sqlalchemy import Column, Integer, String
    from app.db.base import Base

    class Repository(Base):
        __tablename__ = "repositories"

        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)

Tables can later be created using:

    Base.metadata.create_all(bind=engine)
"""

from sqlalchemy.orm import declarative_base
import sys

from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


try:
    logger.info("Initializing SQLAlchemy declarative base")

    Base = declarative_base()

    logger.info("SQLAlchemy Base initialized successfully")

except Exception as e:
    logger.error("Failed to initialize SQLAlchemy Base")
    raise CustomException(e, sys)