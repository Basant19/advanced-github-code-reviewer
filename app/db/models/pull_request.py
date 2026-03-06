"""
Pull Request Model

This module defines the SQLAlchemy ORM model for GitHub Pull Requests
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

Example Record
--------------
id: 1
repo_id: 3
pr_number: 24
title: "Fix authentication bug"
author: "developer123"
branch: "feature/auth-fix"
status: "open"

Usage
-----
    from app.db.models.pull_request import PullRequest
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base


class PullRequest(Base):
    """
    SQLAlchemy ORM model representing a GitHub Pull Request.

    Attributes
    ----------
    id : int
        Primary key identifier.

    repo_id : int
        Foreign key referencing the repository.

    pr_number : int
        Pull request number in GitHub.

    title : str
        Title of the pull request.

    author : str
        GitHub username who created the PR.

    branch : str
        Source branch of the pull request.

    status : str
        Current status of the pull request (open, closed, merged).

    created_at : datetime
        Timestamp when the PR record was created in the system.

    updated_at : datetime
        Timestamp when the PR record was last updated.
    """

    __tablename__ = "pull_requests"

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

    # Relationship to Repository
    repository = relationship("Repository", backref="pull_requests")

    def __repr__(self):
        """
        Return readable representation of the pull request.
        Useful for debugging and logging.
        """
        return f"<PullRequest(repo_id={self.repo_id}, pr_number={self.pr_number})>"