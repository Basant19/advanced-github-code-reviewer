"""
app/services/review_service.py

Review Orchestration Service
----------------------------

This module is responsible for orchestrating the full lifecycle of
an AI-based Pull Request review.

Responsibilities
----------------
1. Trigger LangGraph review workflow
2. Persist review results in database
3. Store workflow execution steps
4. Handle HITL (Human-in-the-loop) interruptions
5. Post results to GitHub

Architecture
------------
FastAPI Route
      │
      ▼
Review Service
      │
      ▼
LangGraph Workflow
      │
      ▼
Database + GitHub API

Error Handling
--------------
All unexpected errors are wrapped inside CustomException so that the
API layer can convert them into HTTP responses.
"""

import sys
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.workflow import review_graph
from app.graph.state import build_initial_state
from app.mcp.github_client import GitHubClient
from app.db.models.repository import Repository
from app.db.models.pull_request import PullRequest
from app.db.models.review import Review
from app.db.models.review_step import ReviewStep
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# GraphInterrupt import (LangGraph compatibility)
# ─────────────────────────────────────────────
try:
    from langgraph.errors import GraphInterrupt
except ImportError:
    try:
        from langgraph.types import GraphInterrupt
    except ImportError:
        class GraphInterrupt(Exception):
            pass


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

async def _get_or_create_repository(
    db: AsyncSession,
    owner: str,
    repo: str
) -> Repository:
    """Fetch repository or create it if missing."""

    try:
        result = await db.execute(
            select(Repository).where(
                Repository.owner == owner,
                Repository.name == repo,
            )
        )

        repository = result.scalar_one_or_none()

        if repository:
            return repository

        repository = Repository(
            owner=owner,
            name=repo,
            url=f"https://github.com/{owner}/{repo}",
            default_branch="main",
        )

        db.add(repository)
        await db.commit()
        await db.refresh(repository)

        logger.info(
            "[review_service] Repository created | %s/%s",
            owner,
            repo,
        )

        return repository

    except Exception as e:
        await db.rollback()
        logger.error(
            "[review_service] Repository creation failed",
            exc_info=True
        )
        raise CustomException(str(e), sys)


async def _get_or_create_pull_request(
    db: AsyncSession,
    repository: Repository,
    pr_number: int,
    metadata: Dict,
) -> PullRequest:
    """Fetch PR record or create stub."""

    try:
        result = await db.execute(
            select(PullRequest).where(
                PullRequest.repo_id == repository.id,
                PullRequest.pr_number == pr_number,
            )
        )

        pr = result.scalar_one_or_none()

        if pr:
            if metadata.get("title"):
                pr.title = metadata["title"]

            if metadata.get("state"):
                pr.status = metadata["state"]

            await db.commit()
            return pr

        pr = PullRequest(
            repo_id=repository.id,
            pr_number=pr_number,
            title=metadata.get("title", "Untitled PR"),
            author=metadata.get("author", "unknown"),
            branch=metadata.get("head_branch", "unknown"),
            status=metadata.get("state", "open"),
        )

        db.add(pr)
        await db.commit()
        await db.refresh(pr)

        logger.info(
            "[review_service] PullRequest created | #%s %s",
            pr_number,
            pr.title,
        )

        return pr

    except Exception as e:
        await db.rollback()
        logger.error(
            "[review_service] PullRequest operation failed",
            exc_info=True
        )
        raise CustomException(str(e), sys)


async def _persist_review_steps(
    db: AsyncSession,
    review: Review,
    final_state: Dict,
) -> None:
    """Persist workflow steps."""

    now = datetime.now(timezone.utc)

    try:
        steps = [
            ReviewStep(
                review_id=review.id,
                step_name="fetch_diff",
                status="completed",
                input_data=json.dumps({"pr_number": final_state.get("pr_number")}),
                output_data=json.dumps({
                    "files_changed": len(final_state.get("files", [])),
                    "diff_length": len(final_state.get("diff", "")),
                }),
                created_at=now,
            ),
            ReviewStep(
                review_id=review.id,
                step_name="analyze_code",
                status="completed",
                input_data=json.dumps({"diff_length": len(final_state.get("diff", ""))}),
                output_data=json.dumps({
                    "issues": final_state.get("issues", []),
                    "suggestions": final_state.get("suggestions", []),
                }),
                created_at=now,
            ),
            ReviewStep(
                review_id=review.id,
                step_name="verdict",
                status="completed",
                input_data=json.dumps({}),
                output_data=json.dumps({
                    "verdict": final_state.get("verdict")
                }),
                created_at=now,
            ),
        ]

        db.add_all(steps)
        await db.commit()

        logger.info(
            "[review_service] %d workflow steps stored | review_id=%s",
            len(steps),
            review.id,
        )

    except Exception as e:
        await db.rollback()
        logger.error(
            "[review_service] Step persistence failed",
            exc_info=True
        )
        raise CustomException(str(e), sys)


# ─────────────────────────────────────────────
# Core Service Functions
# ─────────────────────────────────────────────

async def trigger_review(
    owner: str,
    repo: str,
    pr_number: int,
    db: AsyncSession,
) -> Review:
    """
    Trigger a full PR review workflow.

    Returns
    -------
    Review
        Persisted review object.
    """

    logger.info(
        "[review_service] Trigger review | %s/%s#%s",
        owner,
        repo,
        pr_number,
    )

    try:

        initial_state = build_initial_state(owner, repo, pr_number)

        repository = await _get_or_create_repository(db, owner, repo)

        stub_pr = await _get_or_create_pull_request(
            db,
            repository,
            pr_number,
            {}
        )

        thread_id = str(uuid.uuid4())

        review = Review(
            pull_request_id=stub_pr.id,
            reviewer="ai",
            status="running",
            thread_id=thread_id,
        )

        db.add(review)
        await db.commit()
        await db.refresh(review)

        logger.info(
            "[review_service] Review started | id=%s thread=%s",
            review.id,
            thread_id,
        )

        config = {"configurable": {"thread_id": thread_id}}

        final_state = await review_graph.ainvoke(
            initial_state,
            config=config
        )

        pull_request = await _get_or_create_pull_request(
            db,
            repository,
            pr_number,
            final_state.get("metadata", {}),
        )

        review.pull_request_id = pull_request.id

        await _persist_review_steps(db, review, final_state)

        review.verdict = final_state.get("verdict")
        review.summary = final_state.get("summary")
        review.status = "completed"

        await db.commit()
        await db.refresh(review)

        logger.info(
            "[review_service] Review completed | id=%s verdict=%s",
            review.id,
            review.verdict,
        )

        if review.verdict != "HUMAN_REJECTED":

            try:
                github_client = GitHubClient()

                github_client.post_review_comment(
                    owner,
                    repo,
                    pr_number,
                    final_state.get("summary", ""),
                )

                logger.info(
                    "[review_service] GitHub comment posted"
                )

            except Exception:
                logger.warning(
                    "[review_service] GitHub comment failed",
                    exc_info=True,
                )

        return review

    except GraphInterrupt:

        logger.info(
            "[review_service] HITL interrupt | review_id=%s",
            review.id,
        )

        review.status = "pending_hitl"
        await db.commit()
        await db.refresh(review)

        return review

    except Exception as e:

        await db.rollback()

        try:
            review.status = "failed"
            await db.commit()
        except Exception:
            pass

        logger.error(
            "[review_service] Review execution failed",
            exc_info=True,
        )

        raise CustomException(str(e), sys)


# ─────────────────────────────────────────────
# Query Services
# ─────────────────────────────────────────────

async def get_review(
    review_id: int,
    db: AsyncSession,
) -> Review:
    """Retrieve a single review."""

    try:
        result = await db.execute(
            select(Review).where(Review.id == review_id)
        )

        review = result.scalar_one_or_none()

        if not review:
            raise CustomException(
                f"Review {review_id} not found",
                sys
            )

        return review

    except Exception as e:
        logger.error("[review_service] get_review failed", exc_info=True)
        raise CustomException(str(e), sys)


async def list_reviews(
    owner: str,
    repo: str,
    db: AsyncSession,
) -> List[Review]:
    """List all reviews for a repository."""

    try:
        result = await db.execute(
            select(Repository).where(
                Repository.owner == owner,
                Repository.name == repo,
            )
        )

        repository = result.scalar_one_or_none()

        if not repository:
            return []

        result = await db.execute(
            select(Review)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .where(PullRequest.repo_id == repository.id)
            .order_by(Review.created_at.desc())
        )

        return result.scalars().all()

    except Exception as e:
        logger.error("[review_service] list_reviews failed", exc_info=True)
        raise CustomException(str(e), sys)


async def list_all_reviews(db: AsyncSession) -> List[Dict]:
    """Return all reviews across repositories."""

    try:

        result = await db.execute(
            select(Review, PullRequest, Repository)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .join(Repository, PullRequest.repo_id == Repository.id)
            .order_by(Review.created_at.desc())
        )

        rows = result.all()

        reviews = []

        for review, pr, repo in rows:

            reviews.append({
                "id": review.id,
                "status": review.status,
                "verdict": review.verdict,
                "summary": review.summary,
                "thread_id": review.thread_id,
                "created_at": review.created_at.isoformat() if review.created_at else None,
                "repo": f"{repo.owner}/{repo.name}",
                "pr_number": pr.pr_number,
                "title": pr.title,
                "author": pr.author,
                "branch": pr.branch,
            })

        return reviews

    except Exception as e:
        logger.error("[review_service] list_all_reviews failed", exc_info=True)
        raise CustomException(str(e), sys)