"""
Repository Model

Defines the SQLAlchemy ORM model for GitHub repositories
connected to the Advanced GitHub Code Reviewer platform.

Purpose
-------
The Repository table enables multi-repository support and
stores metadata required for automated pull request reviews.

Each record represents a GitHub repository monitored by
the platform.

Table
-----
repositories
"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from app.db.base import Base


class Repository(Base):
    """
    SQLAlchemy ORM model representing a GitHub repository.

    Attributes
    ----------
    id : int
        Primary key identifier.

    name : str
        Name of the repository.

    owner : str
        GitHub username or organization that owns the repository.

    url : str
        Full GitHub repository URL.

    default_branch : str
        Default branch used by the repository.

    created_at : datetime
        Timestamp when the repository was added to the system.

    updated_at : datetime
        Timestamp when repository metadata was last updated.
    """

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(255), nullable=False, index=True)

    owner = Column(String(255), nullable=False, index=True)

    url = Column(String(500), nullable=False, unique=True)

    default_branch = Column(String(100), default="main")

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now()
    )

    def __repr__(self):
        """
        Return readable representation of the repository.
        Useful for debugging and logging.
        """
        return f"<Repository(owner='{self.owner}', name='{self.name}')>"