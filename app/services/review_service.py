"""
app/services/review_service.py

Review Orchestration Service
------------------------------
The single entry point for triggering, persisting, and retrieving PR reviews.

Responsibilities:
    1. trigger_review()  — run the full LangGraph workflow, persist results,
                           post the summary comment back to GitHub
    2. get_review()      — fetch a single review with its steps from DB
    3. list_reviews()    — list all reviews for a repository

DB models used (from Commit 1):
    Repository   — registered GitHub repos
    PullRequest  — PR metadata per repo
    Review       — one review run per PR (status, verdict, summary)
    ReviewStep   — individual node outputs (fetch, analyze, reflect, verdict)

Flow inside trigger_review():
    ┌─────────────────────────────────────────────────────────┐
    │ 1. get_or_create Repository record                      │
    │ 2. get_or_create PullRequest record                     │
    │ 3. create Review record  (status="running")             │
    │ 4. run_review()  →  LangGraph executes all 4 nodes      │
    │ 5. persist ReviewStep for each stage                    │
    │ 6. update Review  (status="completed", verdict, summary)│
    │ 7. GitHubClient.post_review_comment()                   │
    └─────────────────────────────────────────────────────────┘
    On any failure → update Review status="failed", re-raise

Status values for Review:
    "pending"   — not started (reserved for future queuing)
    "running"   — workflow in progress
    "completed" — verdict produced and comment posted
    "failed"    — unrecoverable error occurred
"""

import sys
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.graph.workflow import run_review
from app.mcp.github_client import GitHubClient
from app.db.models.repository import Repository
from app.db.models.pull_request import PullRequest
from app.db.models.review import Review
from app.db.models.review_step import ReviewStep
from app.core.exceptions import CustomException
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_repository(
    db: Session, owner: str, repo: str
) -> Repository:
    """
    Fetches existing Repository record or creates a new one.
    Commits the new record so it has a valid ID for FK references.
    """
    record = (
        db.query(Repository)
        .filter_by(owner=owner, name=repo)
        .first()
    )
    if record:
        return record

    record = Repository(
        owner=owner,
        name=repo,
        full_name=f"{owner}/{repo}",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(f"[review_service] Created Repository record — {owner}/{repo}")
    return record


def _get_or_create_pull_request(
    db: Session,
    repository: Repository,
    pr_number: int,
    metadata: dict,
) -> PullRequest:
    """
    Fetches existing PullRequest record or creates one from the metadata
    returned by fetch_diff_node.
    """
    record = (
        db.query(PullRequest)
        .filter_by(repository_id=repository.id, number=pr_number)
        .first()
    )
    if record:
        # Update title/state in case they changed
        record.title  = metadata.get("title",  record.title)
        record.state  = metadata.get("state",  record.state)
        db.commit()
        return record

    record = PullRequest(
        repository_id=repository.id,
        number=pr_number,
        title=metadata.get("title",       ""),
        author=metadata.get("author",     ""),
        description=metadata.get("description", ""),
        base_branch=metadata.get("base_branch", ""),
        head_branch=metadata.get("head_branch", ""),
        state=metadata.get("state",       "open"),
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        f"[review_service] Created PullRequest record — "
        f"#{pr_number} '{metadata.get('title', '')}'"
    )
    return record


def _persist_review_steps(
    db: Session, review: Review, final_state: dict
) -> None:
    """
    Creates one ReviewStep record for each logical stage of the workflow.
    Steps are named to match the node that produced them.
    """
    steps = [
        ReviewStep(
            review_id=review.id,
            step_name="fetch_diff",
            input_data={"pr_number": final_state.get("pr_number")},
            output_data={
                "files_changed": len(final_state.get("files", [])),
                "diff_length":   len(final_state.get("diff",  "")),
            },
            created_at=datetime.now(timezone.utc),
        ),
        ReviewStep(
            review_id=review.id,
            step_name="analyze_code",
            input_data={"diff_length": len(final_state.get("diff", ""))},
            output_data={
                "issues":      final_state.get("issues",      []),
                "suggestions": final_state.get("suggestions", []),
                "repo_context_used": bool(final_state.get("repo_context")),
            },
            created_at=datetime.now(timezone.utc),
        ),
        ReviewStep(
            review_id=review.id,
            step_name="reflect",
            input_data={},
            output_data={
                "reflection_count": final_state.get("reflection_count", 0),
                "final_issues":     final_state.get("issues",      []),
                "final_suggestions": final_state.get("suggestions", []),
            },
            created_at=datetime.now(timezone.utc),
        ),
        ReviewStep(
            review_id=review.id,
            step_name="verdict",
            input_data={
                "issues_count":      len(final_state.get("issues",      [])),
                "suggestions_count": len(final_state.get("suggestions", [])),
            },
            output_data={
                "verdict": final_state.get("verdict", ""),
            },
            created_at=datetime.now(timezone.utc),
        ),
    ]

    db.bulk_save_objects(steps)
    db.commit()
    logger.info(
        f"[review_service] Persisted {len(steps)} ReviewStep records "
        f"for review_id={review.id}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def trigger_review(
    owner:     str,
    repo:      str,
    pr_number: int,
    db:        Session,
) -> Review:
    """
    Runs the complete PR review pipeline:
        1. Ensures Repository + PullRequest records exist in DB
        2. Creates a Review record with status="running"
        3. Executes the LangGraph workflow (all 4 nodes + reflection loop)
        4. Persists ReviewStep records for each stage
        5. Updates Review with verdict, summary, status="completed"
        6. Posts the review summary as a GitHub PR comment

    Args:
        owner:     GitHub repository owner login
        repo:      GitHub repository name
        pr_number: Pull request number to review
        db:        SQLAlchemy session (injected by FastAPI dependency)

    Returns:
        The completed Review ORM object.

    Raises:
        CustomException on any unrecoverable error. The Review record is
        updated to status="failed" before the exception propagates.
    """
    logger.info(
        f"[trigger_review] Starting — {owner}/{repo}#{pr_number}"
    )

    # ── Step 1: ensure repo record exists ────────────────────────────────
    repository = _get_or_create_repository(db, owner, repo)

    # ── Step 2: create review record (status=running) ────────────────────
    review = Review(
        repository_id=repository.id,
        pr_number=pr_number,
        status="running",
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    logger.info(
        f"[trigger_review] Review record created — id={review.id}, "
        f"status=running"
    )

    try:
        # ── Step 3: run LangGraph workflow ────────────────────────────────
        final_state = run_review(owner, repo, pr_number)

        # ── Step 4: get_or_create pull request record (needs metadata) ───
        pull_request = _get_or_create_pull_request(
            db, repository, pr_number, final_state.get("metadata", {})
        )
        review.pull_request_id = pull_request.id

        # ── Step 5: persist review steps ─────────────────────────────────
        _persist_review_steps(db, review, final_state)

        # ── Step 6: update review with verdict + summary ──────────────────
        review.verdict     = final_state.get("verdict", "")
        review.summary     = final_state.get("summary", "")
        review.status      = "completed"
        review.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(review)

        logger.info(
            f"[trigger_review] Review completed — "
            f"id={review.id}, verdict={review.verdict}"
        )

        # ── Step 7: post comment to GitHub ────────────────────────────────
        try:
            github_client = GitHubClient()
            github_client.post_review_comment(
                owner, repo, pr_number, final_state["summary"]
            )
            logger.info(
                f"[trigger_review] Review comment posted to "
                f"{owner}/{repo}#{pr_number}"
            )
        except Exception as gh_err:
            # GitHub comment failure must NOT roll back the completed review
            logger.warning(
                f"[trigger_review] Failed to post GitHub comment "
                f"(review still saved): {gh_err}"
            )

        return review

    except Exception as e:
        # Mark review as failed so the caller / UI can surface it
        try:
            review.status      = "failed"
            review.completed_at = datetime.now(timezone.utc)
            db.commit()
        except Exception:
            pass   # DB might also be down — best effort

        logger.error(
            f"[trigger_review] Review failed — id={review.id}: {e}"
        )

        if isinstance(e, CustomException):
            raise
        raise CustomException(str(e), sys)


def get_review(review_id: int, db: Session) -> Review:
    """
    Fetches a single Review by ID, including its ReviewStep records.

    Args:
        review_id: Primary key of the Review record
        db:        SQLAlchemy session

    Returns:
        Review ORM object with .steps relationship loaded.

    Raises:
        CustomException if review not found or DB error.
    """
    logger.info(f"[get_review] Fetching review_id={review_id}")

    try:
        review = db.query(Review).filter_by(id=review_id).first()

        if not review:
            raise ValueError(f"Review id={review_id} not found")

        return review

    except ValueError as e:
        logger.warning(f"[get_review] {e}")
        raise CustomException(str(e), sys)
    except Exception as e:
        logger.error(f"[get_review] DB error: {e}")
        raise CustomException(str(e), sys)


def list_reviews(owner: str, repo: str, db: Session) -> list[Review]:
    """
    Lists all Review records for a given repository, ordered by most recent.

    Args:
        owner: GitHub repository owner
        repo:  GitHub repository name
        db:    SQLAlchemy session

    Returns:
        List of Review ORM objects (may be empty).

    Raises:
        CustomException on DB error.
    """
    logger.info(f"[list_reviews] Fetching reviews for {owner}/{repo}")

    try:
        repository = (
            db.query(Repository)
            .filter_by(owner=owner, name=repo)
            .first()
        )

        if not repository:
            logger.info(
                f"[list_reviews] No repository record found for {owner}/{repo}"
            )
            return []

        reviews = (
            db.query(Review)
            .filter_by(repository_id=repository.id)
            .order_by(Review.created_at.desc())
            .all()
        )

        logger.info(
            f"[list_reviews] Found {len(reviews)} review(s) for {owner}/{repo}"
        )
        return reviews

    except Exception as e:
        logger.error(f"[list_reviews] DB error: {e}")
        raise CustomException(str(e), sys)