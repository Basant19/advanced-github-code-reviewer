"""
app/services/chat_service.py

Chat Service — P5 Production
------------------------------
P4: Message persistence — add, retrieve, clear.
P5: Gemini reply with review context, rate limit guard.

Rate Limit Guard
----------------
process_chat_message() checks the timestamp of the last assistant message
before calling Gemini. If the last reply was less than CHAT_MIN_INTERVAL_S
seconds ago, it returns a friendly wait message without consuming quota.
This prevents users from hitting the 20 RPD free tier limit on chat alone.

Review Context Loading
----------------------
_load_review_context() queries Review directly by thread_id — the UUID
stored in Review.thread_id by review_service.trigger_review(). This avoids
navigating pull_request and repository relationships which may or may not
be eagerly loaded.

P4 Methods (working)
--------------------
_get_or_create_thread()  — get or create Thread DB record
add_message()            — persist message to DB
get_thread_messages()    — fetch all messages oldest first
clear_thread()           — delete all messages for a thread

P5 Methods (working)
--------------------
process_chat_message()   — rate-limited Gemini reply with review context
_load_review_context()   — load Review record by thread_id
_build_system_prompt()   — build context-aware system prompt
"""

import sys
from datetime import datetime, timedelta,timezone
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.models.message import Message
from app.db.models.thread import Thread
from app.db.models.review import Review
from app.core.logger import get_logger
from app.core.exceptions import CustomException
from sqlalchemy.orm import joinedload, selectinload
from app.db.models.pull_request import PullRequest  

logger = get_logger(__name__)


# ── Rate Limit Config ─────────────────────────────────────────────────────────

CHAT_MIN_INTERVAL_S: int = 5
"""
Minimum seconds between consecutive Gemini chat replies for the same thread.
Prevents quota exhaustion when users send messages in rapid succession.
Free tier: 20 RPD on gemini-2.5-flash-lite.
Each review uses 2-4 calls, leaving ~10-16 for chat per day.
At 5s minimum interval this allows bursts but blocks spam.
"""


# ── Response DTO ──────────────────────────────────────────────────────────────

class ChatResponse:
    """Simple DTO returned by get_thread_messages()."""

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


# ── Service ───────────────────────────────────────────────────────────────────

class ChatService:
    """
    Chat message persistence and Gemini reply service.

    Parameters
    ----------
    db : AsyncSession
        Active async database session from FastAPI get_db() dependency.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── P4 — Thread Management ────────────────────────────────────────────────

    async def _get_or_create_thread(self, thread_id: str) -> Thread:
        """
        Fetch Thread by thread_id UUID or create a new one.

        Thread.title is NOT NULL — always set a default value on creation.
        thread_id is the LangGraph UUID stored in Review.thread_id.
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
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(thread)
        await self.db.commit()
        await self.db.refresh(thread)

        logger.info(
            "[chat_service] Thread created — thread_id=%s db_id=%d",
            thread_id, thread.id,
        )
        return thread

    # ── P4 — Message Persistence ──────────────────────────────────────────────

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> Message:
        """
        Persist a message to the database.

        Creates Thread automatically if it does not exist.

        Parameters
        ----------
        thread_id : str  LangGraph UUID
        role      : str  "user" | "assistant" | "system"
        content   : str  message text
        """
        try:
            thread = await self._get_or_create_thread(thread_id)

            message = Message(
                thread_id=thread.id,
                role=role,
                content=content,
                created_at=datetime.now(timezone.utc),
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
        Fetch all messages for a thread ordered oldest first.

        Returns empty list if thread has no messages.
        Raises CustomException if thread not found.
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
                "[chat_service] Thread cleared — thread_id=%s", thread_id
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

    # ── P5 — Rate Limit Guard ─────────────────────────────────────────────────

    async def _check_rate_limit(
        self,
        thread_id: str,
        history: List[ChatResponse],
    ) -> Optional[str]:
        """
        Check if a Gemini call should be skipped due to rate limiting.

        Looks at the most recent assistant message timestamp. If it was
        less than CHAT_MIN_INTERVAL_S seconds ago, returns a wait message.
        Returns None if the call should proceed.

        Parameters
        ----------
        thread_id : str
        history   : List[ChatResponse]  — current thread messages

        Returns
        -------
        str or None
            Wait message string if rate limited, None if OK to proceed.
        """
        assistant_messages = [m for m in history if m.role == "assistant"]

        if not assistant_messages:
            return None  # No previous reply — proceed

        last_reply = assistant_messages[-1]
        if not last_reply.created_at:
            return None

        last_reply_dt = last_reply.created_at
        if not isinstance(last_reply_dt, datetime):
            return None

        elapsed = (datetime.now(timezone.utc) - last_reply_dt).total_seconds()

        if elapsed < CHAT_MIN_INTERVAL_S:
            wait = int(CHAT_MIN_INTERVAL_S - elapsed) + 1
            logger.info(
                "[chat_service] Rate limit guard triggered — "
                "thread_id=%s elapsed=%.1fs wait=%ds",
                thread_id, elapsed, wait,
            )
            return (
                f"⏳ Please wait {wait} second(s) before sending another message."
            )

        return None  # OK to proceed

    # ── P5 — Review Context Loader ────────────────────────────────────────────
    async def _load_review_context(
        self,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the Review record linked to this thread_id with full PR and Repo metadata.
        """
        try:
            # We need to jump two levels deep to get the Repo name and owner
            result = await self.db.execute(
                select(Review)
                .options(
                    joinedload(Review.pull_request).options(
                        joinedload(PullRequest.repository)
                    )
                )
                .where(Review.thread_id == thread_id)
            )
            review = result.scalar_one_or_none()

            if not review or not review.pull_request:
                logger.warning(f"[chat_service] No review context found for thread_id={thread_id}")
                return None

            pr = review.pull_request
            repo = pr.repository

            # This dictionary now perfectly matches what _build_system_prompt expects
            return {
                "pr_number": pr.pr_number,      # From PullRequest model
                "repo":      repo.name,         # From Repository model
                "owner":     repo.owner,        # From Repository model
                "verdict":   review.verdict or "pending",
                "summary":   review.summary or "",
                "status":    review.status,
            }

        except Exception as e:
            logger.warning(
                f"[chat_service] _load_review_context failed — "
                f"thread_id={thread_id} error={str(e)} — continuing without context"
            )
            return None

    # ── P5 — System Prompt Builder ────────────────────────────────────────────

    def _build_system_prompt(
        self,
        review_ctx: Optional[Dict[str, Any]],
        history: List[ChatResponse],
    ) -> str:
        """
        Build context-aware system prompt for Gemini.

        Injects review verdict, summary, and recent conversation history.
        Summary is truncated to 800 chars to keep prompt size reasonable.
        History is limited to last 6 messages (3 exchanges).
        """
        base = (
            "You are an expert AI code reviewer assistant. "
            "You help developers understand AI-generated code review findings "
            "and answer questions about pull requests.\n\n"
            "Be concise, specific, and reference the actual code issues "
            "when answering. If you don't know something, say so clearly."
        )

        if not review_ctx:
            return base

        context_block = (
            f"\n\n--- Review Context ---\n"
            f"Repository: {review_ctx.get('owner', '')}/{review_ctx.get('repo', 'unknown')}\n"
            f"PR Number:  #{review_ctx.get('pr_number', '?')}\n"
            f"Verdict:    {review_ctx.get('verdict', 'unknown')}\n"
            f"Status:     {review_ctx.get('status', 'unknown')}\n"
        )

        summary = review_ctx.get("summary", "")
        if summary:
            context_block += f"\nReview Summary:\n{summary[:800]}\n"

        # Last 6 messages = last 3 exchanges
        if history:
            history_block = "\n--- Conversation History ---\n"
            for msg in history[-6:]:
                role_label = "User" if msg.role == "user" else "Assistant"
                history_block += f"{role_label}: {msg.content[:300]}\n"
            context_block += history_block

        return base + context_block

    # ── P5 — Gemini Reply ─────────────────────────────────────────────────────

    async def process_chat_message(
        self,
        thread_id: str,
        user_message: str,
    ) -> str:
        """
        Store user message, call Gemini with review context, store and return reply.

        Flow
        ----
        1. Load review context linked to this thread_id
        2. Fetch conversation history
        3. Check rate limit — return wait message if too soon
        4. Build system prompt with review findings injected
        5. Call Gemini via safe_llm_invoke()
        6. Persist user message and assistant reply
        7. Return reply text

        Rate Limit
        ----------
        Returns a friendly wait message if called within CHAT_MIN_INTERVAL_S
        seconds of the previous assistant reply. Does NOT consume quota.

        Parameters
        ----------
        thread_id    : str  LangGraph UUID from Review.thread_id
        user_message : str  message text from the user

        Returns
        -------
        str  Gemini reply, rate limit message, or error message
        """
        logger.info(
            "[chat_service] process_chat_message — thread_id=%s",
            thread_id,
        )

        # ── 1. Load review context ────────────────────────────────────────────
        review_ctx = await self._load_review_context(thread_id)

        # ── 2. Fetch conversation history ─────────────────────────────────────
        try:
            history = await self.get_thread_messages(thread_id)
        except CustomException:
            history = []

        # ── 3. Rate limit guard ───────────────────────────────────────────────
        wait_message = await self._check_rate_limit(thread_id, history)
        if wait_message:
            # Persist user message but return wait reply without calling Gemini
            await self.add_message(thread_id, "user", user_message)
            await self.add_message(thread_id, "assistant", wait_message)
            return wait_message

        # ── 4. Build system prompt ────────────────────────────────────────────
        system_prompt = self._build_system_prompt(review_ctx, history)

        # ── 5. Call Gemini ────────────────────────────────────────────────────
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            from app.graph.nodes import (
                safe_llm_invoke,
                FREE_TIER_EXHAUSTED,
                LLM_ERROR,
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            result = await safe_llm_invoke(messages)

        except Exception as e:
            logger.exception(
                "[chat_service] process_chat_message: "
                "LLM invocation failed — thread_id=%s error=%s",
                thread_id, str(e),
            )
            result = "LLM_ERROR"

        # ── 6. Handle LLM failures ────────────────────────────────────────────
        if result == FREE_TIER_EXHAUSTED:
            reply = (
                "⚠️ The AI quota is temporarily exhausted. "
                "Please try again in a few minutes."
            )
        elif result == LLM_ERROR or result == "LLM_ERROR":
            reply = (
                "⚠️ I encountered an error generating a response. "
                "Please try again."
            )
        else:
            reply = result

        # ── 7. Persist and return ─────────────────────────────────────────────
        await self.add_message(thread_id, "user", user_message)
        await self.add_message(thread_id, "assistant", reply)

        logger.info(
            "[chat_service] process_chat_message complete — "
            "thread_id=%s reply_chars=%d",
            thread_id, len(reply),
        )

        return reply