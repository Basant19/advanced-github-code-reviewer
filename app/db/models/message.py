"""
Message Model

This module defines the SQLAlchemy ORM model responsible for
storing chat messages within a conversation thread.

Purpose
-------
The Message table stores individual messages exchanged
between users and the AI reviewer.

Each message belongs to a Thread and represents a single
communication unit in a conversation.

This enables the system to support:

• Resume chat functionality
• AI + human collaboration
• Review discussions
• Agent reasoning visibility
• Conversation history storage

Relationships
-------------
Thread
    └── Message

Table Name
----------
messages

Example Record
--------------
id: 10
thread_id: 3
role: "user"
content: "Why did the AI request changes?"
created_at: "2026-03-07"

Usage
-----
    from app.db.models.message import Message
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

    Attributes
    ----------
    id : int
        Primary key identifier for the message.

    thread_id : int
        Foreign key referencing the associated thread.

    role : str
        Role of the message sender.

        Possible values:
        - user
        - ai
        - system

    content : str
        Text content of the message.

    created_at : datetime
        Timestamp when the message was created.
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
            backref="message_list"
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