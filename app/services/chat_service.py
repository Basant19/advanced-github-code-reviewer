"""
Chat Service Layer

Handles business logic for chat messages.
"""

from typing import List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models.message import Message
from schemas.chat_schema import MessageCreate, ChatResponse
from app.core.logger import get_logger


logger = get_logger(__name__)


class ChatService:
    """
    Service class responsible for chat message operations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_message(self, message_data: MessageCreate) -> ChatResponse:
        """
        Create a new chat message.
        """

        logger.info("Creating new chat message")

        try:
            message = Message(
                thread_id=message_data.thread_id,
                role=message_data.role,
                content=message_data.content,
            )

            self.db.add(message)
            await self.db.commit()
            await self.db.refresh(message)

            logger.info(f"Message created successfully: {message.id}")

            return ChatResponse(
                thread_id=message.thread_id,
                message_id=message.id,
                role=message.role,
                content=message.content,
            )

        except Exception as e:
            logger.error("Failed to create message", exc_info=True)
            await self.db.rollback()
            raise e

    async def get_thread_messages(self, thread_id: int) -> List[ChatResponse]:
        """
        Retrieve all messages for a thread.
        """

        logger.info(f"Fetching messages for thread: {thread_id}")

        try:
            stmt = (
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at)
            )

            result = await self.db.execute(stmt)
            messages = result.scalars().all()

            logger.info(f"Fetched {len(messages)} messages")

            return [
                ChatResponse(
                    thread_id=m.thread_id,
                    message_id=m.id,
                    role=m.role,
                    content=m.content,
                )
                for m in messages
            ]

        except Exception:
            logger.error("Failed to fetch thread messages", exc_info=True)
            raise

    async def delete_message(self, message_id: int) -> bool:
        """
        Delete a specific message.
        """

        logger.info(f"Deleting message: {message_id}")

        try:
            stmt = select(Message).where(Message.id == message_id)

            result = await self.db.execute(stmt)
            message = result.scalar_one_or_none()

            if not message:
                logger.warning(f"Message not found: {message_id}")
                return False

            await self.db.delete(message)
            await self.db.commit()

            logger.info(f"Message deleted: {message_id}")
            return True

        except Exception:
            logger.error("Failed to delete message", exc_info=True)
            await self.db.rollback()
            raise