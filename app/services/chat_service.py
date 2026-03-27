"""
app/services/chat_service.py

Chat Service — P4 Stub / P5 Foundation
----------------------------------------
P4: Basic message persistence (add, retrieve, clear).
P5: Add Gemini LLM responses + LangGraph thread resume + streaming.

The service works with thread_id strings directly — the thread_id is
the LangGraph MemorySaver thread_id (UUID) from the Review record.
This means chat threads are naturally scoped to reviews.

P4 Methods (working now)
------------------------
add_message()         — persist a user or assistant message
get_thread_messages() — fetch all messages for a thread
clear_thread()        — delete all messages for a thread

P5 Methods (stub — raises NotImplementedError)
----------------------------------------------
process_chat_message() — send to Gemini, stream response, persist
"""

import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.models.message import Message
from app.db.models.thread import Thread
from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)


class ChatResponse:
    """Simple response object for get_thread_messages()."""
    def __init__(
        self,
        thread_id: str,
        message_id: int,
        role: str,
        content: str,
        created_at,
    ):
        self.thread_id  = thread_id
        self.message_id = message_id
        self.role       = role
        self.content    = content
        self.created_at = created_at


class ChatService:
    """
    Chat message persistence service.

    Parameters
    ----------
    db : AsyncSession
        Active async database session from get_db() dependency.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── P4 — Working Methods ──────────────────────────────────────────────────

    async def _get_or_create_thread(self, thread_id: str) -> Thread:
        """
        Fetch existing Thread by thread_id or create a new one.

        thread_id is the LangGraph UUID from Review.thread_id.
        Thread record is a lightweight DB anchor for messages.
        """
        result = await self.db.execute(
            select(Thread).where(Thread.thread_id == thread_id)
        )
        thread = result.scalar_one_or_none()

        if thread:
            return thread

        thread = Thread(
            thread_id=thread_id,
            title=f"Chat {thread_id[:8]}", 
           
        )

        self.db.add(thread)
        await self.db.commit()
        await self.db.refresh(thread)

        logger.info(
            "[chat_service] Thread created — thread_id=%s db_id=%d",
            thread_id, thread.id,
        )
        return thread

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> Message:
        """
        Persist a message to the database.

        Creates the Thread record if it does not exist yet.
        thread_id is the LangGraph UUID from Review.thread_id.

        Parameters
        ----------
        thread_id : str  LangGraph thread UUID
        role      : str  "user" | "assistant" | "system"
        content   : str  message text

        Returns
        -------
        Message  persisted message record
        """
        try:
            thread = await self._get_or_create_thread(thread_id)

            message = Message(
                thread_id=thread.id,
                role=role,
                content=content,
                created_at=datetime.utcnow(),
            )
            self.db.add(message)
            await self.db.commit()
            await self.db.refresh(message)

            logger.info(
                "[chat_service] Message added — "
                "thread_id=%s role=%s message_id=%d",
                thread_id, role, message.id,
            )
            return message

        except Exception as e:
            logger.exception(
                "[chat_service] add_message failed — thread_id=%s error=%s",
                thread_id, str(e),
            )
            await self.db.rollback()
            raise CustomException(str(e), sys)

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 50,
    ) -> List[ChatResponse]:
        """
        Fetch all messages for a thread, oldest first.

        Returns empty list if thread has no messages.
        Returns 404-equivalent CustomException if thread not found.

        Parameters
        ----------
        thread_id : str  LangGraph thread UUID
        limit     : int  max messages to return (default 50)

        Returns
        -------
        List[ChatResponse]
        """
        try:
            result = await self.db.execute(
                select(Thread).where(Thread.thread_id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                raise CustomException(
                    f"Thread {thread_id} not found", sys
                )

            result = await self.db.execute(
                select(Message)
                .where(Message.thread_id == thread.id)
                .order_by(Message.created_at)
                .limit(limit)
            )
            messages = result.scalars().all()

            logger.info(
                "[chat_service] get_thread_messages — "
                "thread_id=%s count=%d",
                thread_id, len(messages),
            )

            return [
                ChatResponse(
                    thread_id=thread_id,
                    message_id=m.id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                )
                for m in messages
            ]

        except CustomException:
            raise
        except Exception as e:
            logger.exception(
                "[chat_service] get_thread_messages failed — "
                "thread_id=%s error=%s",
                thread_id, str(e),
            )
            raise CustomException(str(e), sys)

    async def clear_thread(self, thread_id: str) -> None:
        """
        Delete all messages for a thread.

        Raises CustomException if thread not found.

        Parameters
        ----------
        thread_id : str  LangGraph thread UUID
        """
        try:
            result = await self.db.execute(
                select(Thread).where(Thread.thread_id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                raise CustomException(
                    f"Thread {thread_id} not found", sys
                )

            await self.db.execute(
                delete(Message).where(Message.thread_id == thread.id)
            )
            await self.db.commit()

            logger.info(
                "[chat_service] Thread cleared — thread_id=%s", thread_id,
            )

        except CustomException:
            raise
        except Exception as e:
            logger.exception(
                "[chat_service] clear_thread failed — "
                "thread_id=%s error=%s",
                thread_id, str(e),
            )
            await self.db.rollback()
            raise CustomException(str(e), sys)

    # ── P5 — Stubs ────────────────────────────────────────────────────────────

    async def process_chat_message(
        self,
        thread_id: str,
        user_message: str,
    ):
        """
        P5: Send user message to Gemini and stream the response.

        Will use LangGraph Command(resume=user_message) to inject the
        message into the review graph thread, giving Gemini access to:
        - Review findings (issues, suggestions, verdict)
        - Repository context from ChromaDB (P4 RAG)
        - Full conversation history via MemorySaver checkpointing

        Not implemented yet — raises NotImplementedError.
        """
        raise NotImplementedError(
            "process_chat_message() is planned for P5. "
            "It will send the message to Gemini via LangGraph and stream the response."
        )