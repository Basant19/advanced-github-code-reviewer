"""
Review Model
Location: E:\advanced-github-code-reviewer\app\db\models\review.py

ORM model for storing the lifecycle of a code review.
Supports 'pending_hitl' status to facilitate Human-In-The-Loop interrupts.
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
    Represents a single execution of the AI review graph.
    
    Attributes:
        status: running, pending_hitl, completed, failed, rejected.
        thread_id: Used by LangGraph MemorySaver to resume interrupted states.
    """
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    
    # FK to PullRequest
    pull_request_id = Column(
        Integer, 
        ForeignKey("pull_requests.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )

    # Metadata
    reviewer = Column(String(50), default="ai_agent", nullable=False)
    
    # HITL-Compatible Statuses: ['running', 'pending_hitl', 'completed', 'failed', 'rejected']
    status = Column(String(50), default="running", nullable=False)
    
    # Results
    verdict = Column(String(50), nullable=True)  # APPROVE, REQUEST_CHANGES, FAILED
    summary = Column(Text, nullable=True)
    
    # LangGraph State Persistence
    thread_id = Column(String(100), nullable=True, index=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    pull_request = relationship("PullRequest", back_populates="reviews")
    steps = relationship(
        "ReviewStep", 
        back_populates="review", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self):
        return f"<Review(id={self.id}, pr={self.pull_request_id}, status='{self.status}')>"

logger.info("[DB] Review model loaded")