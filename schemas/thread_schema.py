"""
schemas/thread_schema.py

Thread and Message Pydantic Schemas — P5
------------------------------------------
Request and response schemas for the per-PR chat thread endpoints.

STATUS: Stub — planned for P5 (Resume Chat UI in Streamlit).

Planned Schemas (P5)
--------------------
ThreadResponse
    id: int
    review_id: int
    thread_id: str        ← LangGraph MemorySaver thread identifier
    created_at: datetime
    message_count: int

MessageRequest
    content: str          ← user message text
    role: str             ← "user" | "assistant"

MessageResponse
    id: int
    thread_id: str
    role: str
    content: str
    created_at: datetime

ChatHistoryResponse
    thread_id: str
    messages: List[MessageResponse]
    total: int

Why This File Exists
--------------------
Consistent with the schema pattern established in P1:
    schemas/review_schema.py     ← P1/P2 review schemas
    schemas/chat_schema.py       ← P1 chat schemas
    schemas/thread_schema.py     ← P5 thread/message schemas (stub)

P5 Implementation Plan
----------------------
1. Define ThreadResponse, MessageRequest, MessageResponse here
2. Wire into app/api/routes/chat.py (currently uses inline schemas)
3. Connect to app/services/chat_service.py (planned P5)
4. Used by streamlit_app/pages/review_chat.py dashboard page
"""

from app.core.logger import get_logger

logger = get_logger(__name__)
logger.debug("thread_schema: stub loaded — full implementation planned for P5")
"""

These are fine as-is. Now fix the one-line Pydantic error in `repos.py` and test the full P4 flow:


POST /repos/index       → index repo
GET  /repos/Basant19/python_tuts/status  → verify indexed
POST /reviews/trigger   → trigger review with RAG context

"""