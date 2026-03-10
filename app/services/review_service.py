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

Flow inside trigger_review():
    ┌─────────────────────────────────────────────────────────┐
    │ 1. get_or_create Repository record                      │
    │ 2. get_or_create PullRequest stub (before workflow)     │
    │ 3. create Review record  (status="running")             │
    │ 4. run_review()  →  LangGraph executes all 4 nodes      │
    │ 5. update PullRequest with real metadata from GitHub    │
    │ 6. persist ReviewStep for each stage                    │
    │ 7. update Review  (status="completed", verdict, summary)│
    │ 8. GitHubClient.post_review_comment()                   │
    └─────────────────────────────────────────────────────────┘
    On any failure → update Review status="failed", re-raise

Column names (aligned to actual DB schema):
    Repository  : id, name, owner, url, default_branch, created_at, updated_at
    PullRequest : id, repo_id, pr_number, title, author, branch,
                  status, created_at, updated_at
    Review      : id, pull_request_id, reviewer, status, verdict,
                  summary, created_at, updated_at
    ReviewStep  : id, review_id, step_name, status, input_data,
                  output_data, logs, created_at
"""

import sys
import json
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
    Fetches existing Repository or creates a new one.

    Repository model columns:
        id, name, owner, url, default_branch, created_at, updated_at
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
        url=f"https://github.com/{owner}/{repo}",  # url is NOT NULL
        default_branch="main",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(f"[review_service] Created Repository — {owner}/{repo}")
    return record


def _get_or_create_pull_request(
    db:         Session,
    repository: Repository,
    pr_number:  int,
    metadata:   dict,
) -> PullRequest:
    """
    Fetches existing PullRequest or creates a new one.

    PullRequest model columns:
        id, repo_id, pr_number, title, author, branch,
        status, created_at, updated_at

    Key corrections vs old code:
        repo_id       not repository_id
        pr_number     not number
        branch        not head_branch (model has single branch column)
        status        not state
    """
    record = (
        db.query(PullRequest)
        .filter_by(repo_id=repository.id, pr_number=pr_number)
        .first()
    )
    if record:
        # Update mutable fields on re-review
        new_title = metadata.get("title")
        new_state = metadata.get("state")
        if new_title:
            record.title = new_title
        if new_state:
            record.status = new_state
        db.commit()
        return record

    record = PullRequest(
        repo_id=repository.id,
        pr_number=pr_number,
        title=metadata.get("title",        "Untitled PR"),
        author=metadata.get("author",      "unknown"),
        branch=metadata.get("head_branch", "unknown"),
        status=metadata.get("state",       "open"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        f"[review_service] Created PullRequest — "
        f"#{pr_number} '{record.title}'"
    )
    return record


def _persist_review_steps(
    db:          Session,
    review:      Review,
    final_state: dict,
) -> None:
    """
    Bulk-inserts 4 ReviewStep records — one per LangGraph node.

    ReviewStep model columns:
        id, review_id, step_name, status, input_data,
        output_data, logs, created_at

    Note: input_data and output_data are TEXT columns — must be JSON strings.
    """
    now = datetime.now(timezone.utc)

    steps = [
        ReviewStep(
            review_id=review.id,
            step_name="fetch_diff",
            status="completed",
            input_data=json.dumps({
                "pr_number": final_state.get("pr_number"),
            }),
            output_data=json.dumps({
                "files_changed": len(final_state.get("files", [])),
                "diff_length":   len(final_state.get("diff",  "")),
            }),
            created_at=now,
        ),
        ReviewStep(
            review_id=review.id,
            step_name="analyze_code",
            status="completed",
            input_data=json.dumps({
                "diff_length": len(final_state.get("diff", "")),
            }),
            output_data=json.dumps({
                "issues":            final_state.get("issues",      []),
                "suggestions":       final_state.get("suggestions", []),
                "repo_context_used": bool(final_state.get("repo_context")),
            }),
            created_at=now,
        ),
        ReviewStep(
            review_id=review.id,
            step_name="reflect",
            status="completed",
            input_data=json.dumps({}),
            output_data=json.dumps({
                "reflection_count":  final_state.get("reflection_count", 0),
                "final_issues":      final_state.get("issues",      []),
                "final_suggestions": final_state.get("suggestions", []),
            }),
            created_at=now,
        ),
        ReviewStep(
            review_id=review.id,
            step_name="verdict",
            status="completed",
            input_data=json.dumps({
                "issues_count":      len(final_state.get("issues",      [])),
                "suggestions_count": len(final_state.get("suggestions", [])),
            }),
            output_data=json.dumps({
                "verdict": final_state.get("verdict", ""),
            }),
            created_at=now,
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
    Runs the complete PR review pipeline and returns the completed Review record.

    Review model columns:
        pull_request_id, reviewer, status, verdict, summary,
        created_at, updated_at

    No repository_id on Review — Review links to PullRequest, not Repository.
    No pr_number on Review — that lives on PullRequest.
    No completed_at on Review — updated_at tracks last modification.

    Raises:
        CustomException on any unrecoverable error.
    """
    logger.info(f"[trigger_review] Starting — {owner}/{repo}#{pr_number}")

    # Step 1 — ensure repository record exists
    repository = _get_or_create_repository(db, owner, repo)

    # Step 2 — create a PullRequest stub so Review FK is satisfied
    # Metadata is empty here — real metadata comes from GitHub after workflow
    stub_pr = _get_or_create_pull_request(db, repository, pr_number, {})

    # Step 3 — create Review in "running" state
    review = Review(
        pull_request_id=stub_pr.id,  # pull_request_id not repository_id
        reviewer="ai",
        status="running",
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    logger.info(
        f"[trigger_review] Review record created — "
        f"id={review.id}, status=running"
    )

    try:
        # Step 4 — run LangGraph workflow (20-60s)
        final_state = run_review(owner, repo, pr_number)

        # Step 5 — update PullRequest with real metadata fetched by workflow
        pull_request = _get_or_create_pull_request(
            db, repository, pr_number,
            final_state.get("metadata", {}),
        )
        review.pull_request_id = pull_request.id

        # Step 6 — persist 4 step records
        _persist_review_steps(db, review, final_state)

        # Step 7 — mark review completed
        review.verdict = final_state.get("verdict", "")
        review.summary = final_state.get("summary", "")
        review.status  = "completed"
        # updated_at auto-set by onupdate=func.now()
        db.commit()
        db.refresh(review)

        logger.info(
            f"[trigger_review] Completed — "
            f"id={review.id}, verdict={review.verdict}"
        )

        # Step 8 — post GitHub comment (non-fatal)
        try:
            github_client = GitHubClient()
            github_client.post_review_comment(
                owner, repo, pr_number,
                final_state.get("summary", ""),
            )
            logger.info(
                f"[trigger_review] GitHub comment posted — "
                f"{owner}/{repo}#{pr_number}"
            )
        except Exception as gh_err:
            logger.warning(
                f"[trigger_review] GitHub comment failed "
                f"(review still saved): {gh_err}"
            )

        return review

    except Exception as e:
        # Best-effort status update — don't let this mask the real error
        try:
            review.status = "failed"
            db.commit()
        except Exception:
            pass

        logger.error(f"[trigger_review] Failed — id={review.id}: {e}")

        if isinstance(e, CustomException):
            raise
        raise CustomException(str(e), sys)


def get_review(review_id: int, db: Session) -> Review:
    """
    Fetches a single Review by ID, steps accessible via review.steps.

    Raises:
        CustomException if not found or DB error.
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
    Lists all reviews for a repo ordered by most recent first.

    Review has no direct repo FK — must join through PullRequest.
    Returns empty list if repo has no record, never raises for missing repo.
    """
    logger.info(f"[list_reviews] {owner}/{repo}")

    try:
        repository = (
            db.query(Repository)
            .filter_by(owner=owner, name=repo)
            .first()
        )

        if not repository:
            logger.info(f"[list_reviews] No repository found for {owner}/{repo}")
            return []

        # Review → PullRequest → Repository (two hops, no direct FK)
        reviews = (
            db.query(Review)
            .join(PullRequest, Review.pull_request_id == PullRequest.id)
            .filter(PullRequest.repo_id == repository.id)
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