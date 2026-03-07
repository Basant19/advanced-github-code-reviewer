"""
Thread Model

This module defines the SQLAlchemy ORM model responsible for
storing chat threads within the Advanced GitHub Code Reviewer
platform.

Purpose
-------
A Thread represents a conversation session between users and
the AI code reviewer.

Threads allow users to:

• Resume previous conversations
• Discuss specific pull requests
• Ask follow-up questions about review results
• Provide human feedback to the AI reviewer
• Rename conversations for better organization

Each thread may optionally be linked to a Pull Request so
that discussions remain contextual to a specific code review.

Relationships
-------------
Repository
    └── PullRequest
            └── Thread
                    └── Message

Table Name
----------
threads

Example Record
--------------
id: 3
title: "Discussion about PR #42 review"
pull_request_id: 42

Usage
-----
    from app.db.models.thread import Thread
"""

import sys

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


class Thread(Base):
    """
    SQLAlchemy ORM model representing a conversation thread.

    A thread stores a discussion session between a user and the
    AI code reviewer. Threads can be renamed and resumed later.

    Attributes
    ----------
    id : int
        Primary key identifier for the thread.

    title : str
        User-defined name for the conversation thread.

    pull_request_id : int
        Optional foreign key referencing the related pull request.

    created_at : datetime
        Timestamp when the thread was created.

    updated_at : datetime
        Timestamp when the thread was last updated.
    """

    try:

        __tablename__ = "threads"

        id = Column(
            Integer,
            primary_key=True,
            index=True
        )

        title = Column(
            String(255),
            nullable=False
        )

        pull_request_id = Column(
            Integer,
            ForeignKey("pull_requests.id"),
            nullable=True,
            index=True
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

        # Relationship → Pull Request
        pull_request = relationship(
            "PullRequest",
            backref="threads"
        )

        # Relationship → Messages (defined later)
        messages = relationship(
            "Message",
            backref="thread",
            cascade="all, delete-orphan"
        )

        logger.info("Thread model initialized successfully")

    except Exception as e:
        logger.error("Error while defining Thread model")
        raise CustomException(e, sys)

    def __repr__(self):
        """
        Return readable representation for debugging.
        """
        return f"<Thread(id={self.id}, title='{self.title}')>"