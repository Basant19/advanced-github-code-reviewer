"""
Review Model

Defines the ORM model responsible for storing AI or human
code review results generated for GitHub Pull Requests.

Each review represents a single execution of the review
pipeline on a pull request.

Supports:
• Multiple review runs
• AI review history
• Reflection loops
• Human approvals
• Debugging review pipelines
"""

import sys

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


class Review(Base):
    """
    SQLAlchemy ORM model representing a code review
    execution for a pull request.
    """

    __tablename__ = "reviews"

    try:

        id = Column(Integer, primary_key=True, index=True)

        pull_request_id = Column(
            Integer,
            ForeignKey("pull_requests.id"),
            nullable=False,
            index=True
        )

        reviewer = Column(
            String(50),
            default="ai",
            nullable=False
        )

        status = Column(
            String(50),
            default="pending",
            nullable=False
        )

        verdict = Column(
            String(50),
            nullable=True
        )

        summary = Column(
            Text,
            nullable=True
        )

        created_at = Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False
        )

        updated_at = Column(
            DateTime(timezone=True),
            onupdate=func.now()
        )

        # Relationship → PullRequest
        pull_request = relationship(
            "PullRequest",
            back_populates="reviews"
        )

        # Relationship → ReviewStep
        steps = relationship(
            "ReviewStep",
            back_populates="review",
            cascade="all, delete-orphan"
        )

        logger.info("Review model initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize Review model", exc_info=True)
        raise CustomException(e, sys)

    def __repr__(self):
        return (
            f"<Review(id={self.id}, "
            f"pull_request_id={self.pull_request_id}, "
            f"status={self.status})>"
        )