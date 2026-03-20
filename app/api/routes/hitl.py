"""
hitl.py (Production Hardened - FINAL)

ROLE:
------
Thin API layer for HITL actions.

CRITICAL DESIGN:
---------------
✔ Delegates ALL logic to review_service (single source of truth)
✔ Prevents duplicate graph execution
✔ Prevents inconsistent state bugs
✔ Fully aligned with workflow + interrupt_before

WHY THIS CHANGE:
---------------
❌ Old version directly resumed graph → DUPLICATION
❌ Caused Invalid Review State + double execution

✅ New version:
- Only calls decide_review()
- Service handles everything (graph + DB + validation)
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import decide_review, get_review

logger = get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["hitl"])


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    """
    Optional reviewer input (future use)
    """
    reviewer_note: Optional[str] = Field(default=None, max_length=2000)


class HITLDecisionResponse(BaseModel):
    review_id: int
    decision: str
    verdict: Optional[str]
    message: str


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _handle_error(e: Exception, context: str):
    logger.error(
        f"[HITL] {context}",
        extra={"error": str(e), "type": type(e).__name__},
        exc_info=True,
    )

    if isinstance(e, CustomException):
        msg = str(e).lower()

        if "not found" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))

        if "invalid" in msg:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

        if "checkpoint" in msg:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e))

        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e))

    raise HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Internal server error",
    )


# ─────────────────────────────────────────────
# APPROVE (FIXED)
# ─────────────────────────────────────────────

@router.post("/{review_id}/approve", response_model=HITLDecisionResponse)
async def approve_review(
    review_id: int,
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Approve a pending review.

    FLOW:
    -----
    API → review_service.decide_review → LangGraph resume → DB update
    """

    logger.info(
        "[HITL] Approve request",
        extra={"review_id": review_id},
    )

    try:
        review = await decide_review(
            review_id=review_id,
            decision="approved",
            db=db,
        )

        return HITLDecisionResponse(
            review_id=review.id,
            decision="approved",
            verdict=review.verdict,
            message="Review approved and completed",
        )

    except Exception as e:
        _handle_error(e, "approve_review")


# ─────────────────────────────────────────────
# REJECT (FIXED)
# ─────────────────────────────────────────────

@router.post("/{review_id}/reject", response_model=HITLDecisionResponse)
async def reject_review(
    review_id: int,
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Reject a pending review.
    """

    logger.info(
        "[HITL] Reject request",
        extra={"review_id": review_id},
    )

    try:
        review = await decide_review(
            review_id=review_id,
            decision="rejected",
            db=db,
        )

        return HITLDecisionResponse(
            review_id=review.id,
            decision="rejected",
            verdict=review.verdict,
            message="Review rejected",
        )

    except Exception as e:
        _handle_error(e, "reject_review")


# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

@router.get("/{review_id}/status")
async def get_review_status(
    review_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch current review status (safe read-only).
    """

    logger.info(
        "[HITL] Status request",
        extra={"review_id": review_id},
    )

    try:
        review = await get_review(review_id=review_id, db=db)

        return {
            "review_id": review.id,
            "status": review.status,
            "verdict": review.verdict,
            "thread_id": review.thread_id,
            "created_at": review.created_at,
            "updated_at": review.updated_at,
        }

    except Exception as e:
        _handle_error(e, "get_review_status")