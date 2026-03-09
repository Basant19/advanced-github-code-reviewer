"""
reset_db.py

Database Reset Script
----------------------
Drops all tables and recreates them from the current SQLAlchemy models.

USE WHEN:
    - ORM model was changed (column added/removed/renamed)
    - DB schema is out of sync with models
    - Starting fresh during development

WARNING:
    All data is permanently deleted. Dev environments only.

Usage:
    python reset_db.py
"""

import sys
from app.db.base import Base
from app.db.session import engine
from app.db.models import *  # noqa — ensures all models are registered
from app.core.logger import get_logger

logger = get_logger(__name__)


def reset():
    logger.info("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped.")

    logger.info("Recreating all tables from models...")
    Base.metadata.create_all(bind=engine)
    logger.info("All tables recreated successfully.")
    logger.info("Database reset complete — ready to use.")


if __name__ == "__main__":
    confirm = input("This will DELETE ALL DATA. Type 'yes' to continue: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)
    reset()