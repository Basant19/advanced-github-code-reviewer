"""
app/api/routes/chat.py

Chat API Routes — P4 Stub (P5 will add Gemini responses)
----------------------------------------------------------
Basic message persistence for PR chat threads.
P5 will add LangGraph resume + Gemini streaming responses.

Endpoints:
    POST   /chat/{thread_id}/messages  — store a message
    GET    /chat/{thread_id}/messages  — fetch all messages
    DELETE /chat/{thread_id}           — delete thread messages
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.chat_service import ChatService

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SendMessageRequest(BaseModel):
    content: str
    role:    str = "user"


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:         int
    thread_id:  str
    role:       str
    content:    str
    created_at: Optional[str] = None


class ThreadMessagesResponse(BaseModel):
    thread_id:     str
    message_count: int
    messages:      List[MessageResponse]


class DeleteThreadResponse(BaseModel):
    thread_id: str
    deleted:   bool
    message:   str


# ── Helper ────────────────────────────────────────────────────────────────────

def _handle_error(e: Exception, context: str) -> None:
    msg = str(e).lower()
    if "not found" in msg:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    logger.error("[chat_route] %s error: %s", context, e)
    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal error during {context}",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/{thread_id}/messages",
    status_code=http_status.HTTP_201_CREATED,
    response_model=MessageResponse,
    summary="Send a message in a thread",
)
async def send_message(
    thread_id: str,
    request:   SendMessageRequest,
    db:        AsyncSession = Depends(get_db),
) -> MessageResponse:
    """
    Store a message in the thread.
    Thread is created automatically if it does not exist.
    P5 will add Gemini response generation here.
    """
    logger.info(
        "[chat_route] Send message — thread_id=%s role=%s",
        thread_id, request.role,
    )

    try:
        service = ChatService(db)
        message = await service.add_message(
            thread_id=thread_id,
            role=request.role,
            content=request.content,
        )

        return MessageResponse(
            id=message.id,
            thread_id=thread_id,
            role=message.role,
            content=message.content,
            created_at=str(message.created_at) if message.created_at else None,
        )

    except CustomException as e:
        _handle_error(e, "send_message")


@router.get(
    "/{thread_id}/messages",
    status_code=http_status.HTTP_200_OK,
    response_model=ThreadMessagesResponse,
    summary="Get all messages in a thread",
)
async def get_messages(
    thread_id: str,
    db:        AsyncSession = Depends(get_db),
) -> ThreadMessagesResponse:
    """
    Returns all messages in a thread ordered by creation time.
    Returns empty list if thread has no messages.
    Returns 404 if thread does not exist.
    """
    logger.info("[chat_route] Get messages — thread_id=%s", thread_id)

    try:
        service = ChatService(db)
        messages = await service.get_thread_messages(thread_id=thread_id)

        return ThreadMessagesResponse(
            thread_id=thread_id,
            message_count=len(messages),
            messages=[
                MessageResponse(
                    id=m.message_id,
                    thread_id=thread_id,
                    role=m.role,
                    content=m.content,
                    created_at=str(m.created_at) if m.created_at else None,
                )
                for m in messages
            ],
        )

    except CustomException as e:
        _handle_error(e, "get_messages")


@router.delete(
    "/{thread_id}",
    status_code=http_status.HTTP_200_OK,
    response_model=DeleteThreadResponse,
    summary="Delete all messages in a thread",
)
async def delete_thread(
    thread_id: str,
    db:        AsyncSession = Depends(get_db),
) -> DeleteThreadResponse:
    """
    Deletes all messages in a thread.
    Returns 404 if thread does not exist.
    """
    logger.info("[chat_route] Delete thread — thread_id=%s", thread_id)

    try:
        service = ChatService(db)
        await service.clear_thread(thread_id=thread_id)

        return DeleteThreadResponse(
            thread_id=thread_id,
            deleted=True,
            message=f"Thread '{thread_id}' cleared successfully",
        )

    except CustomException as e:
        _handle_error(e, "delete_thread")