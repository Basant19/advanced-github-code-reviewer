"""
E:\advanced-github-code-reviewer\app\db\models\pull_request.py
Pull Request Model

Defines the SQLAlchemy ORM model for GitHub Pull Requests
tracked by the Advanced GitHub Code Reviewer platform.

Purpose
-------
The PullRequest table stores metadata about pull requests detected
from connected GitHub repositories.

This table acts as the central entity for the automated review workflow.

Each pull request record represents a PR event that the platform
will analyze, review, and potentially comment on.

Relationships
-------------
Repository
    └── PullRequest
            └── Review
                    └── ReviewStep

Table Name
----------
pull_requests
"""

import sys

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


class PullRequest(Base):
    """
    SQLAlchemy ORM model representing a GitHub Pull Request.
    """

    __tablename__ = "pull_requests"

    try:

        id = Column(Integer, primary_key=True, index=True)

        repo_id = Column(
            Integer,
            ForeignKey("repositories.id"),
            nullable=False,
            index=True
        )

        pr_number = Column(Integer, nullable=False)

        title = Column(String(500), nullable=False)

        author = Column(String(255), nullable=False)

        branch = Column(String(255), nullable=False)

        status = Column(String(50), default="open")

        created_at = Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False
        )

        updated_at = Column(
            DateTime(timezone=True),
            onupdate=func.now()
        )

        # Relationship → Repository
        repository = relationship(
            "Repository",
            back_populates="pull_requests"
        )

        # Relationship → Reviews
        reviews = relationship(
            "Review",
            back_populates="pull_request",
            cascade="all, delete-orphan"
        )

        logger.info("PullRequest model initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize PullRequest model", exc_info=True)
        raise CustomException(e, sys)

    def __repr__(self):
        return (
            f"<PullRequest("
            f"repo_id={self.repo_id}, "
            f"pr_number={self.pr_number}, "
            f"status={self.status})>"
        )