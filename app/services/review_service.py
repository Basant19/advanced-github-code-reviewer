"""
review_service.py (PRODUCTION FINAL - HITL ENFORCED)

DESIGN:
✔ ALWAYS HITL (no auto-complete)
✔ Safe GraphInterrupt lifecycle
✔ No double execution
✔ LLM-safe (never crashes system)
✔ Full logging + observability
✔ Async DB safe

FLOW:
trigger_review → runs graph → ALWAYS pauses → status=pending_hitl
decide_review  → resumes graph → completes → status=completed
"""

import sys
import uuid
from typing import List, Any

from sqlalchemy.orm import selectinload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from app.graph.workflow import get_review_graph
from app.graph.state import build_initial_state

from app.db.models.repository import Repository
from app.db.models.pull_request import PullRequest
from app.db.models.review import Review

from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────

async def _get_or_create_repository(db: AsyncSession, owner: str, name: str) -> Repository:
    result = await db.execute(
        select(Repository).where(
            Repository.owner == owner,
            Repository.name == name
        )
    )
    repo = result.scalar_one_or_none()

    if not repo:
        repo = Repository(owner=owner, name=name)
        db.add(repo)
        await db.flush()
        logger.info(f"[DB] Created repository {owner}/{name}")

    return repo


async def _get_or_create_pull_request(
    db: AsyncSession,
    repo: Repository,
    pr_number: int
) -> PullRequest:
    result = await db.execute(
        select(PullRequest).where(
            PullRequest.repo_id == repo.id,
            PullRequest.pr_number == pr_number
        )
    )
    pr = result.scalar_one_or_none()

    if not pr:
        pr = PullRequest(
            repo_id=repo.id,
            pr_number=pr_number,
            title=f"PR #{pr_number}"
        )
        db.add(pr)
        await db.flush()
        logger.info(f"[DB] Created PR #{pr_number}")

    return pr


# ─────────────────────────────────────────────
# READ APIs
# ─────────────────────────────────────────────

async def get_review(review_id: int, db: AsyncSession) -> Review:
    try:
        logger.info(f"[SERVICE] Fetch review {review_id}")

        result = await db.execute(
            select(Review)
            .options(selectinload(Review.steps))
            .where(Review.id == review_id)
        )

        review = result.scalar_one_or_none()

        if not review:
            raise CustomException(f"Review {review_id} not found", sys)

        return review

    except Exception:
        logger.exception("[SERVICE] get_review failed")
        raise


async def list_reviews(owner: str, repo: str, db: AsyncSession) -> List[Review]:
    try:
        logger.info(f"[SERVICE] List reviews for {owner}/{repo}")

        result = await db.execute(
            select(Review)
            .join(PullRequest)
            .join(Repository)
            .where(
                Repository.owner == owner,
                Repository.name == repo
            )
            .order_by(Review.created_at.desc())
        )

        return list(result.scalars().all())

    except Exception:
        logger.exception("[SERVICE] list_reviews failed")
        raise


# ─────────────────────────────────────────────
# TRIGGER REVIEW (START)
# ─────────────────────────────────────────────

async def trigger_review(
    owner: str,
    repo: str,
    pr_number: int,
    db: AsyncSession
) -> Review:
    """
    Starts review and ALWAYS pauses for HITL.
    """

    graph = get_review_graph()

    try:
        logger.info(f"[REVIEW] Trigger start {owner}/{repo}#{pr_number}")

        # ───── DB SETUP ─────
        repo_obj = await _get_or_create_repository(db, owner, repo)
        pr_obj = await _get_or_create_pull_request(db, repo_obj, pr_number)

        review = Review(
            pull_request_id=pr_obj.id,
            status="running",
            thread_id=str(uuid.uuid4())
        )

        db.add(review)
        await db.commit()
        await db.refresh(review)

        logger.info(f"[REVIEW] Created review {review.id}")

        # ───── GRAPH EXECUTION ─────
        state = build_initial_state(owner, repo, pr_number)

        config = {
            "configurable": {
                "thread_id": review.thread_id
            }
        }

        try:
            await graph.ainvoke(state, config=config)

            # 🔥 CRITICAL: If graph finishes → FORCE HITL
            logger.warning("[HITL] No interrupt occurred → forcing HITL")

            review.status = "pending_hitl"

        except GraphInterrupt:
            logger.info(f"[HITL] Review {review.id} paused correctly")
            review.status = "pending_hitl"

        except Exception:
            logger.exception("[REVIEW] Graph execution failed")
            review.status = "failed"

        await db.commit()

        return review

    except Exception as e:
        logger.exception("[REVIEW] Trigger failed")
        raise CustomException(str(e), sys)


# ─────────────────────────────────────────────
# RESUME REVIEW (HUMAN DECISION)
# ─────────────────────────────────────────────

async def decide_review(
    review_id: int,
    decision: str,
    db: AsyncSession
) -> Review:
    """
    Resume graph after human decision.
    """

    graph = get_review_graph()

    try:
        logger.info(f"[HITL] Resume review {review_id} with decision={decision}")

        review = await get_review(review_id, db)

        if review.status != "pending_hitl":
            raise CustomException("Review not in HITL state", sys)

        config = {
            "configurable": {
                "thread_id": review.thread_id
            }
        }

        # 🔥 RESUME GRAPH
        command = Command(
            resume={
                "human_decision": decision
            }
        )

        try:
            await graph.ainvoke(command, config=config)

            review.status = "completed"
            logger.info(f"[REVIEW] Completed {review.id}")

        except Exception:
            logger.exception("[REVIEW] Resume failed")
            review.status = "failed"

        await db.commit()
        return review

    except Exception as e:
        logger.exception("[HITL] Decision failed")
        raise CustomException(str(e), sys)


# ─────────────────────────────────────────────
# GLOBAL DASHBOARD
# ─────────────────────────────────────────────

async def list_all_reviews(db: AsyncSession) -> List[Review]:
    try:
        logger.info("[SERVICE] List all reviews")

        result = await db.execute(
            select(Review)
            .options(
                selectinload(Review.pull_request)
                .selectinload(PullRequest.repository)
            )
            .order_by(Review.created_at.desc())
        )

        return list(result.scalars().all())

    except Exception:
        logger.exception("[SERVICE] list_all_reviews failed")
        raise CustomException("Database error", sys)