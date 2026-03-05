"""
Repository Model

This module defines the Repository database model used to store
information about GitHub repositories connected to the platform.

Purpose
-------
The Repository table enables the system to support multiple
GitHub repositories and track metadata required for automated
code reviews.

Each repository entry represents a GitHub repository that
the platform monitors for Pull Request events.

Responsibilities
---------------
1. Store repository identity information.
2. Maintain repository ownership metadata.
3. Track repository URL and default branch.
4. Enable relationship mapping with Pull Requests and Reviews.

Relationships
-------------
Repository
    ├── PullRequest
    │       └── Review
    │              └── ReviewStep
    │
    └── ChatThread (optional future link)

Table Name
----------
repositories

Example Record
--------------
id: 1
name: advanced-github-code-reviewer
owner: Basant19
url: https://github.com/Basant19/advanced-github-code-reviewer
default_branch: main

Usage
-----
    from app.db.models.repository import Repository
"""

from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
import sys

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


class Repository(Base):
    """
    SQLAlchemy ORM model for GitHub repositories.

    Attributes
    ----------
    id : int
        Primary key for the repository record.

    name : str
        Repository name.

    owner : str
        GitHub repository owner.

    url : str
        Full GitHub repository URL.

    default_branch : str
        Default branch of the repository.

    created_at : datetime
        Timestamp when the repository was registered
        inside the platform.
    """

    try:
        __tablename__ = "repositories"

        id = Column(Integer, primary_key=True, index=True)

        name = Column(String, nullable=False)

        owner = Column(String, nullable=False)

        url = Column(String, nullable=False, unique=True)

        default_branch = Column(String, default="main")

        created_at = Column(
            DateTime,
            default=datetime.utcnow,
            nullable=False
        )

        def __repr__(self):
            """
            Return readable string representation of repository.
            Useful for debugging and logs.
            """
            return f"<Repository(name={self.name}, owner={self.owner})>"

    except Exception as e:
        logger.error("Error while defining Repository model")
        raise CustomException(e, sys)