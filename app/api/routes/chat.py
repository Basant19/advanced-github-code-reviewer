"""
app/api/routes/chat.py

Chat API Routes — P5
---------------------
POST /chat/{thread_id}/messages  — send message, get Gemini reply
GET  /chat/{thread_id}/messages  — fetch full message history
DELETE /chat/{thread_id}         — clear all messages in thread

P5: send_message now calls process_chat_message() which generates
a Gemini reply using review context and returns it to the caller.
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


class SendMessageResponse(BaseModel):
    """
    Response for POST /chat/{thread_id}/messages.
    Returns both the stored user message and the Gemini reply.
    """
    thread_id:      str
    user_message:   str
    reply:          str
    reply_role:     str = "assistant"


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
    response_model=SendMessageResponse,
    summary="Send a message in a thread",
    description=(
        "Sends a user message and returns a Gemini AI reply "
        "generated using the review context for this thread.\n\n"
        "The thread_id is the LangGraph UUID from `Review.thread_id` — "
        "visible in `GET /reviews/id/{id}` or `GET /reviews/{id}/status`."
    ),
)
async def send_message(
    thread_id: str,
    request:   SendMessageRequest,
    db:        AsyncSession = Depends(get_db),
) -> SendMessageResponse:
    """
    Store user message and return Gemini reply.

    The Gemini reply is generated with full review context:
    - PR verdict and summary from the linked Review record
    - Conversation history from this thread
    - System prompt instructing Gemini to act as code review assistant
    """
    logger.info(
        "[chat_route] Send message — thread_id=%s role=%s chars=%d",
        thread_id, request.role, len(request.content),
    )

    try:
        service = ChatService(db)
        reply = await service.process_chat_message(
            thread_id=thread_id,
            user_message=request.content,
        )

        return SendMessageResponse(
            thread_id=thread_id,
            user_message=request.content,
            reply=reply,
            reply_role="assistant",
        )

    except CustomException as e:
        _handle_error(e, "send_message")

    except Exception as e:
        logger.exception(
            "[chat_route] send_message unexpected error — "
            "thread_id=%s error=%s",
            thread_id, str(e),
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during send_message",
        )


@router.get(
    "/{thread_id}/messages",
    status_code=http_status.HTTP_200_OK,
    response_model=ThreadMessagesResponse,
    summary="Get all messages in a thread",
    description=(
        "Returns all messages in the thread ordered oldest first.\n\n"
        "Returns empty messages list if thread has no messages yet. "
        "Returns 404 if thread does not exist."
    ),
)
async def get_messages(
    thread_id: str,
    db:        AsyncSession = Depends(get_db),
) -> ThreadMessagesResponse:
    """Fetch full message history for a thread."""
    logger.info("[chat_route] Get messages — thread_id=%s", thread_id)

    try:
        service  = ChatService(db)
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
    description="Clears all messages. Returns 404 if thread does not exist.",
)
async def delete_thread(
    thread_id: str,
    db:        AsyncSession = Depends(get_db),
) -> DeleteThreadResponse:
    """Delete all messages for a thread."""
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

