"""
Chat Schemas

This module defines Pydantic schemas used for handling chat
requests and responses within the Advanced GitHub Code Reviewer
platform.

Purpose
-------
These schemas validate API request and response data related
to chat interactions between users and the AI reviewer.

The chat system allows users to:

• Ask questions about code review results
• Discuss pull request changes
• Resume previous conversations
• Provide feedback to the AI reviewer

Schemas Defined
---------------
ChatRequest
    Schema used when a user sends a chat message.

ChatResponse
    Schema returned when the AI responds to a message.

ThreadCreate
    Schema used to create a new conversation thread.

MessageCreate
    Schema used to store a new message in the database.

Example Request
---------------
POST /chat

{
    "thread_id": 3,
    "message": "Why did the AI request changes?"
}

Example Response
----------------
{
    "thread_id": 3,
    "message_id": 10,
    "role": "ai",
    "content": "The linter detected unused imports."
}
"""

import sys
from typing import Optional
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.core.exceptions import CustomException


logger = get_logger(__name__)


try:

    class ThreadCreate(BaseModel):
        """
        Schema for creating a new conversation thread.

        Attributes
        ----------
        title : str
            Name of the conversation thread.

        pull_request_id : Optional[int]
            Optional pull request associated with the thread.
        """

        title: str = Field(
            ...,
            description="Title of the conversation thread"
        )

        pull_request_id: Optional[int] = Field(
            None,
            description="Associated pull request ID"
        )


    class MessageCreate(BaseModel):
        """
        Schema used to store a new message in the system.

        Attributes
        ----------
        thread_id : int
            ID of the thread where the message belongs.

        role : str
            Role of the sender (user, ai, system).

        content : str
            Message content.
        """

        thread_id: int = Field(
            ...,
            description="Thread ID where message belongs"
        )

        role: str = Field(
            ...,
            description="Role of the sender (user, ai, system)"
        )

        content: str = Field(
            ...,
            description="Text content of the message"
        )


    class ChatRequest(BaseModel):
        """
        Schema representing a chat request sent by the user.

        Attributes
        ----------
        thread_id : int
            Thread where the message should be added.

        message : str
            User's message text.
        """

        thread_id: int = Field(
            ...,
            description="Conversation thread ID"
        )

        message: str = Field(
            ...,
            description="User message content"
        )


    class ChatResponse(BaseModel):
        """
        Schema returned by the API after processing
        a chat request.

        Attributes
        ----------
        thread_id : int
            Thread where the message belongs.

        message_id : int
            Unique identifier for the stored message.

        role : str
            Role of the sender (ai).

        content : str
            AI-generated response.
        """

        thread_id: int = Field(
            ...,
            description="Thread ID"
        )

        message_id: int = Field(
            ...,
            description="Unique message identifier"
        )

        role: str = Field(
            ...,
            description="Sender role"
        )

        content: str = Field(
            ...,
            description="AI response message"
        )


    logger.info("Chat schemas loaded successfully")


except Exception as e:
    logger.error("Error while defining chat schemas")
    raise CustomException(e, sys)