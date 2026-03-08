"""
Message Model

This module defines the SQLAlchemy ORM model responsible for
storing chat messages within a conversation thread.
"""

import sys

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


class Message(Base):
    """
    SQLAlchemy ORM model representing a single message
    inside a conversation thread.
    """

    try:

        __tablename__ = "messages"

        id = Column(
            Integer,
            primary_key=True,
            index=True
        )

        thread_id = Column(
            Integer,
            ForeignKey("threads.id"),
            nullable=False,
            index=True
        )

        role = Column(
            String(50),
            nullable=False
        )

        content = Column(
            Text,
            nullable=False
        )

        created_at = Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False
        )

        # Relationship → Thread
        thread = relationship(
            "Thread",
            back_populates="messages"
        )

        logger.info("Message model initialized successfully")

    except Exception as e:
        logger.error("Error while defining Message model")
        raise CustomException(e, sys)

    def __repr__(self):
        """
        Return readable representation for debugging.
        """
        return f"<Message(id={self.id}, role={self.role})>"