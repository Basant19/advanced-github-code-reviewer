"""
app/api/routes/hitl.py

P3 — Human-in-the-Loop (HITL) approval endpoints.

Endpoints:
    POST /reviews/{review_id}/approve
    POST /reviews/{review_id}/reject
    GET  /reviews/{review_id}/status

Flow:
    1. Admin calls one of these endpoints (from Streamlit dashboard or curl).
    2. Route validates the review exists and is in 'pending_hitl' status.
    3. Sets human_decision in graph state via LangGraph thread resume.
    4. Resumes the suspended graph thread — graph continues from hitl_node
       to verdict_node, which reads human_decision and builds final output.
    5. Updates Review.status in PostgreSQL.

LangGraph resume pattern:
    The graph was suspended by interrupt() inside hitl_node.
    To resume, we invoke the compiled graph again with the same thread_id
    (via config={"configurable": {"thread_id": ...}}) and pass the human
    decision via graph.update_state() then ainvoke(None).

Note on imports
---------------
`select` and `Review` are imported at MODULE LEVEL (not inside functions).
This is intentional: it allows tests to patch `app.api.routes.hitl.select`
and `app.api.routes.hitl.Review` without the patch being ignored because
the names were already resolved inside a local scope.

    # In tests:
    with patch("app.api.routes.hitl.select", mock_select):
        ...
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.db.models import Review          # ← correct path: app/db/models.py
from app.graph.workflow import review_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reviews", tags=["hitl"])


# ── Request / Response schemas ────────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    """Optional body for approve/reject — allows a reviewer comment."""
    reviewer_note: Optional[str] = None


class HITLDecisionResponse(BaseModel):
    """Returned after processing an approve or reject action."""
    review_id: int
    decision: str           # 'approved' | 'rejected'
    verdict: Optional[str] = None    # graph verdict after resumption
    message: str


# ── Helper: load review from DB ───────────────────────────────────────────────

async def _get_pending_review(review_id: int, db: AsyncSession):
    """
    Load a Review record and verify it is awaiting HITL decision.
    Raises 404 if not found, 409 if not in correct state.

    `select` and `Review` are module-level names — patchable in tests via:
        patch("app.api.routes.hitl.select", mock_select)
    """
    result = await db.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found.",
        )

    if review.status != "pending_hitl":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Review {review_id} is not awaiting HITL approval. "
                f"Current status: '{review.status}'. "
                f"Expected: 'pending_hitl'."
            ),
        )

    return review


async def _resume_graph(thread_id: str, human_decision: str) -> dict:
    """
    Resume the suspended LangGraph thread with the human decision.

    Steps:
        1. Update the checkpointed state — inject human_decision.
        2. Invoke the graph with None input to continue from checkpoint.
        3. Return the final state dict.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Step 1: inject human_decision into checkpointed state
        review_graph.update_state(
            config,
            {"human_decision": human_decision},
        )
        logger.info(
            "_resume_graph: injected human_decision='%s' into thread '%s'",
            human_decision, thread_id,
        )

        # Step 2: resume — ainvoke(None) continues from the interrupt() point
        final_state = await review_graph.ainvoke(None, config=config)
        logger.info(
            "_resume_graph: graph completed. verdict='%s'",
            final_state.get("verdict"),
        )
        return final_state

    except Exception as exc:
        logger.error(
            "_resume_graph: failed to resume thread '%s': %s",
            thread_id, exc, exc_info=True,
        )
        raise CustomException(
            f"Failed to resume review graph: {exc}"
        ) from exc


# ── POST /reviews/{review_id}/approve ────────────────────────────────────────

@router.post(
    "/{review_id}/approve",
    response_model=HITLDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve a pending AI review",
    description=(
        "Resume a suspended review graph with human_decision='approved'. "
        "The graph continues to verdict_node and posts the review comment to GitHub."
    ),
)
async def approve_review(
    review_id: int,
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Human approves the AI-generated review.

    The graph resumes from interrupt(), verdict_node sees human_decision='approved',
    produces APPROVE or REQUEST_CHANGES verdict, and posts the GitHub comment.
    """
    review = await _get_pending_review(review_id, db)
    thread_id = review.thread_id

    if not thread_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Review {review_id} has no thread_id — cannot resume graph.",
        )

    logger.info(
        "approve_review: review_id=%d thread_id='%s' note='%s'",
        review_id, thread_id, body.reviewer_note,
    )

    try:
        final_state = await _resume_graph(thread_id, "approved")

        review.status  = "completed"
        review.verdict = final_state.get("verdict")
        review.summary = final_state.get("summary", "")
        await db.commit()
        await db.refresh(review)

        logger.info(
            "approve_review: review_id=%d completed. verdict='%s'",
            review_id, review.verdict,
        )

        return HITLDecisionResponse(
            review_id=review_id,
            decision="approved",
            verdict=review.verdict,
            message=(
                f"Review approved. Verdict: {review.verdict}. "
                "GitHub comment posted (if approved verdict)."
            ),
        )

    except CustomException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ── POST /reviews/{review_id}/reject ─────────────────────────────────────────

@router.post(
    "/{review_id}/reject",
    response_model=HITLDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Reject a pending AI review",
    description=(
        "Resume a suspended review graph with human_decision='rejected'. "
        "The graph continues to verdict_node which sets verdict=HUMAN_REJECTED "
        "and skips posting any GitHub comment."
    ),
)
async def reject_review(
    review_id: int,
    body: HITLDecisionRequest = HITLDecisionRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Human rejects the AI-generated review.

    verdict_node sees human_decision='rejected', sets verdict=HUMAN_REJECTED,
    and does NOT post to GitHub. Review is saved with rejected status.
    """
    review = await _get_pending_review(review_id, db)
    thread_id = review.thread_id

    if not thread_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Review {review_id} has no thread_id — cannot resume graph.",
        )

    logger.info(
        "reject_review: review_id=%d thread_id='%s' note='%s'",
        review_id, thread_id, body.reviewer_note,
    )

    try:
        final_state = await _resume_graph(thread_id, "rejected")

        review.status  = "rejected"
        review.verdict = "HUMAN_REJECTED"
        review.summary = final_state.get("summary", "Rejected by human reviewer.")
        await db.commit()
        await db.refresh(review)

        logger.info(
            "reject_review: review_id=%d marked as HUMAN_REJECTED", review_id,
        )

        return HITLDecisionResponse(
            review_id=review_id,
            decision="rejected",
            verdict="HUMAN_REJECTED",
            message="Review rejected. No comment posted to GitHub.",
        )

    except CustomException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


# ── GET /reviews/{review_id}/status ──────────────────────────────────────────

@router.get(
    "/{review_id}/status",
    summary="Get current status of a review (for dashboard polling)",
)
async def get_review_status(
    review_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the current status of a review.
    Streamlit dashboard polls this to detect when a review enters pending_hitl.

    `select` and `Review` are module-level — patchable in tests via:
        patch("app.api.routes.hitl.select", mock_select)
    """
    result = await db.execute(select(Review).where(Review.id == review_id))
    review = result.scalar_one_or_none()

    if review is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found.",
        )

    return {
        "review_id":  review_id,
        "status":     review.status,
        "verdict":    review.verdict,
        "thread_id":  review.thread_id,
        "created_at": review.created_at,
        "updated_at": review.updated_at,
    }