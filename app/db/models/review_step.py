"""
ReviewStep Model
Location: E:\advanced-github-code-reviewer\app\db\models\review_step.py

Stores granular execution traces of each node in the LangGraph.
Crucial for debugging AI reflections and audit trails.
"""

import sys
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base import Base
from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)

class ReviewStep(Base):
    """
    Stores input/output for specific graph nodes (e.g., 'analyze_code', 'lint_check').
    """
    __tablename__ = "review_steps"

    id = Column(Integer, primary_key=True, index=True)
    
    review_id = Column(
        Integer, 
        ForeignKey("reviews.id", ondelete="CASCADE"), 
        nullable=False, 
        index=True
    )

    # Step Metadata
    step_name = Column(String(100), nullable=False)  # Node name from workflow.py
    status = Column(String(50), default="completed", nullable=False)

    # Data Blobs
    input_data = Column(Text, nullable=True)   # JSON string of input state
    output_data = Column(Text, nullable=True)  # JSON string of output or changes
    logs = Column(Text, nullable=True)         # Detailed node-level logs

    created_at = Column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc), 
        nullable=False
    )

    # Relationship back to parent Review
    review = relationship("Review", back_populates="steps")

    def __repr__(self):
        return f"<ReviewStep(review_id={self.review_id}, name='{self.step_name}', status='{self.status}')>"

logger.info("[DB] ReviewStep model loaded")