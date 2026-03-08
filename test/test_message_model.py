"""
Test suite for Message model.
"""

import pytest

from app.db.models.message import Message


def test_message_model_fields():
    """
    Test that Message model fields are correctly defined.
    """
    message = Message(
        thread_id=1,
        role="user",
        content="Test message"
    )

    assert message.thread_id == 1
    assert message.role == "user"
    assert message.content == "Test message"


def test_message_repr():
    """
    Test __repr__ method of Message model.
    """
    message = Message(
        id=5,
        thread_id=2,
        role="ai",
        content="AI response"
    )

    assert "<Message" in repr(message)