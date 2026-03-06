"""
SQLAlchemy Declarative Base

Defines the Base class used by all ORM models.

Responsibilities
----------------
1. Provides central metadata registry
2. Registers all database models
3. Enables table creation and migrations
"""

from sqlalchemy.orm import declarative_base
import sys

from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)


try:
    logger.info("Initializing SQLAlchemy declarative base")

    # Create base class for ORM models
    Base = declarative_base()

    logger.info("SQLAlchemy Base initialized successfully")

    # IMPORTANT:
    # Import all models here so SQLAlchemy registers them
    # with the Base metadata.
    from app.db.models.repository import Repository

    logger.info("Database models registered successfully")

except Exception as e:
    logger.error("Failed to initialize SQLAlchemy Base")
    raise CustomException(e, sys)