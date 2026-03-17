#E:\advanced-github-code-reviewer\app\api\routes\review.py
from typing import Optional, List, Any, NoReturn, Literal
from datetime import datetime
import json
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import (
    trigger_review,
    get_review,
    list_reviews,
    decide_review,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/reviews", tags=["reviews"])

# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class TriggerReviewRequest(BaseModel):
    owner: str
    repo: str
    pr_number: int

class ReviewDecisionRequest(BaseModel):
    """Request body for human-in-the-loop decisions."""
    decision: Literal["approved", "rejected"]

class ReviewStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    step_name: str
    input_data: Optional[Any] = None 
    output_data: Optional[Any] = None

class ReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    verdict: Optional[str] = None
    summary: Optional[str] = None
    # Changed from str to datetime for validation and sorting
    created_at: Optional[datetime] = None

class ReviewDetailResponse(ReviewResponse):
    steps: List[ReviewStepResponse] = Field(default_factory=list)

class TriggerReviewResponse(BaseModel):
    review_id: int
    status: str
    verdict: Optional[str] = None
    message: str

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _handle_service_error(e: Exception, context: str) -> NoReturn:
    """
    Explicitly handles service errors. NoReturn ensures the caller 
    knows an exception is always raised.
    """
    log_data = {"context": context, "error": str(e)}
    
    if isinstance(e, CustomException):
        logger.error(f"[review_route] Service error: {context}", extra=log_data, exc_info=True)
        status_code = http_status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else http_status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(status_code=status_code, detail=str(e))

    logger.error(f"[review_route] Unexpected error: {context}", extra=log_data, exc_info=True)
    raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

def _safe_parse_json(data: Any) -> Any:
    """
    Safely parses JSON. 
    Guarantees a dictionary return for UI consistency.
    Handles legacy stringified Python objects.
    """
    if data is None:
        return {}
        
    if isinstance(data, str):
        # 1. Attempt JSON parsing
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed} # Wrap lists for consistent dict return
            return {"value": parsed}
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Specific check for stringified Python objects (Legacy Data)
        if data.startswith("SandboxResult("):
            return {"type": "sandbox_result", "raw": data}

        # 3. Handle raw text (like Git Diffs)
        return {"type": "text", "content": data}
            
    # 4. If already a dict, return as is
    if isinstance(data, dict):
        return data
        
    # 5. Final fallback
    return {"raw": str(data)}
# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.post("/trigger", response_model=TriggerReviewResponse, status_code=http_status.HTTP_202_ACCEPTED)
async def trigger_review_route(request: TriggerReviewRequest, db: AsyncSession = Depends(get_db)):
    logger.info("Triggering review", extra={"repo": f"{request.owner}/{request.repo}", "pr": request.pr_number})
    try:
        review = await trigger_review(
            owner=request.owner, repo=request.repo, pr_number=request.pr_number, db=db
        )
        message = "Review suspended at HITL gate" if review.status == "pending_hitl" else "Review completed"
        return TriggerReviewResponse(review_id=review.id, status=review.status, verdict=review.verdict, message=message)
    except Exception as e:
        _handle_service_error(e, "trigger_review_route")

@router.get("/repo/{owner}/{repo}", response_model=List[ReviewResponse])
async def list_reviews_route(owner: str, repo: str, db: AsyncSession = Depends(get_db)):
    logger.info("Listing reviews", extra={"owner": owner, "repo": repo})
    try:
        reviews = await list_reviews(owner=owner, repo=repo, db=db)
        return [ReviewResponse.model_validate(r) for r in reviews]
    except Exception as e:
        _handle_service_error(e, "list_reviews_route")

@router.get("/id/{review_id}", response_model=ReviewDetailResponse)
async def get_review_route(review_id: int, db: AsyncSession = Depends(get_db)):
    logger.info("Fetching review details", extra={"review_id": review_id})
    try:
        review = await get_review(review_id=review_id, db=db)
        steps = [
            ReviewStepResponse(
                id=s.id,
                step_name=s.step_name,
                input_data=_safe_parse_json(s.input_data),
                output_data=_safe_parse_json(s.output_data),
            ) for s in (review.steps or [])
        ]
        
        # Note: model_validate handles the datetime conversion automatically now
        result = ReviewDetailResponse.model_validate(review)
        result.steps = steps
        return result
    except Exception as e:
        _handle_service_error(e, "get_review_route")

@router.post("/id/{review_id}/decision", response_model=ReviewResponse)
async def decide_review_route(
    review_id: int, 
    req: ReviewDecisionRequest, 
    db: AsyncSession = Depends(get_db)
):
    """Consolidated endpoint for Approving or Rejecting a review."""
    logger.info("Processing HITL decision", extra={"review_id": review_id, "decision": req.decision})
    try:
        review = await decide_review(
            review_id=review_id,
            decision=req.decision,
            db=db,
        )
        return ReviewResponse.model_validate(review)
    except Exception as e:
        _handle_service_error(e, "decide_review_route")