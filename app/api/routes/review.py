"""
app/api/routes/review.py

Review API Routes
------------------
REST endpoints for triggering and retrieving PR reviews.

Endpoints:
    POST /reviews/trigger              — manually trigger a review (no webhook needed)
    GET  /reviews/{owner}/{repo}       — list all reviews for a repository
    GET  /reviews/{review_id}          — get a single review with its steps

All routes delegate to review_service.py — no business logic lives here.

Error handling:
    CustomException from the service layer is caught and converted to
    appropriate HTTP responses (404 for not found, 500 for unexpected errors).

Import chain (no circular imports):
    review.py  →  review_service.py  →  graph/workflow.py
                                     →  db/models/*
                                     →  mcp/github_client.py
"""

import sys
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import (
    trigger_review,
    get_review,
    list_reviews,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ── Request / Response schemas ────────────────────────────────────────────────
# Defined inline to avoid circular imports with schemas/ package.

class TriggerReviewRequest(BaseModel):
    owner:     str
    repo:      str
    pr_number: int


class ReviewStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:          int
    step_name:   str
    input_data:  Optional[dict] = None
    output_data: Optional[dict] = None


class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           int
    pr_number:    int
    status:       str
    verdict:      Optional[str] = None
    summary:      Optional[str] = None
    created_at:   Optional[str] = None
    completed_at: Optional[str] = None


class ReviewDetailResponse(ReviewResponse):
    model_config = ConfigDict(from_attributes=True)

    steps: list[ReviewStepResponse] = []


class TriggerReviewResponse(BaseModel):
    review_id: int
    status:    str
    verdict:   Optional[str] = None
    message:   str


# ── Helper ────────────────────────────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> None:
    """
    Converts CustomException to HTTP responses.
    "not found" in message → 404, anything else → 500.
    """
    message = str(e).lower()
    if "not found" in message:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    logger.error(f"[review_route] {context}: {e}")
    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal error during {context}",
    )


# ── Routes ────────────────────────────────────────────────────────────────────
# Order matters: /trigger must come before /{owner}/{repo} so FastAPI
# does not try to match "trigger" as an owner path parameter.

@router.post(
    "/trigger",
    status_code=http_status.HTTP_202_ACCEPTED,
    response_model=TriggerReviewResponse,
    summary="Manually trigger a PR review",
)
def trigger_review_route(
    request: TriggerReviewRequest,
    db:      Session = Depends(get_db),
) -> TriggerReviewResponse:
    """
    Manually triggers a PR review without needing a GitHub webhook.
    Useful for testing and re-reviewing existing PRs.

    Runs synchronously — blocks until review completes.
    Use the webhook endpoint for async background execution.
    """
    logger.info(
        f"[review_route] Manual trigger — "
        f"{request.owner}/{request.repo}#{request.pr_number}"
    )

    try:
        review = trigger_review(
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            db=db,
        )

        return TriggerReviewResponse(
            review_id=review.id,
            status=review.status,
            verdict=review.verdict,
            message=(
                f"Review completed for "
                f"{request.owner}/{request.repo}#{request.pr_number}"
            ),
        )

    except CustomException as e:
        _handle_service_error(e, "trigger_review")


@router.get(
    "/{owner}/{repo}",
    status_code=http_status.HTTP_200_OK,
    response_model=list[ReviewResponse],
    summary="List all reviews for a repository",
)
def list_reviews_route(
    owner: str,
    repo:  str,
    db:    Session = Depends(get_db),
) -> list[ReviewResponse]:
    """
    Returns all reviews for a given repository, ordered most recent first.
    Returns an empty list if the repository has no reviews yet.
    """
    logger.info(f"[review_route] List reviews — {owner}/{repo}")

    try:
        reviews = list_reviews(owner=owner, repo=repo, db=db)

        return [
            ReviewResponse(
                id=r.id,
                pr_number=r.pr_number,
                status=r.status,
                verdict=r.verdict,
                summary=r.summary,
                created_at=str(r.created_at)     if r.created_at   else None,
                completed_at=str(r.completed_at) if r.completed_at else None,
            )
            for r in reviews
        ]

    except CustomException as e:
        _handle_service_error(e, "list_reviews")


@router.get(
    "/{review_id}",
    status_code=http_status.HTTP_200_OK,
    response_model=ReviewDetailResponse,
    summary="Get a single review with its steps",
)
def get_review_route(
    review_id: int,
    db:        Session = Depends(get_db),
) -> ReviewDetailResponse:
    """
    Returns a single review by ID, including all ReviewStep records
    showing what each LangGraph node produced.
    """
    logger.info(f"[review_route] Get review — id={review_id}")

    try:
        review = get_review(review_id=review_id, db=db)

        steps = [
            ReviewStepResponse(
                id=s.id,
                step_name=s.step_name,
                input_data=s.input_data,
                output_data=s.output_data,
            )
            for s in (review.steps or [])
        ]

        return ReviewDetailResponse(
            id=review.id,
            pr_number=review.pr_number,
            status=review.status,
            verdict=review.verdict,
            summary=review.summary,
            created_at=str(review.created_at)     if review.created_at   else None,
            completed_at=str(review.completed_at) if review.completed_at else None,
            steps=steps,
        )

    except CustomException as e:
        _handle_service_error(e, "get_review")