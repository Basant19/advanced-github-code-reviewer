"""
SQLAlchemy Declarative Base

Defines the Base class used by all ORM models.

Responsibilities
----------------
1. Provides central metadata registry
2. Base class for all ORM models
3. Enables table creation and migrations
"""

import sys
from sqlalchemy.orm import declarative_base

from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)

try:
    logger.info("Initializing SQLAlchemy declarative base")

    # Base class for ORM models
    Base = declarative_base()

    logger.info("SQLAlchemy Base initialized successfully")

except Exception as e:
    logger.error("Failed to initialize SQLAlchemy Base", exc_info=True)
    raise CustomException(e, sys)