"""
app/api/routes/webhook.py

GitHub Webhook Route
---------------------
Receives POST /webhook/github from GitHub when PR events occur.

Security:
    Every request is verified against the X-Hub-Signature-256 header
    using HMAC-SHA256 with GITHUB_WEBHOOK_SECRET. Requests that fail
    verification are rejected with 403 before any processing.

Handled events:
    pull_request — actions: "opened", "synchronize", "reopened"

Ignored events:
    Everything else returns 200 with {"status": "ignored"} so GitHub
    doesn't treat them as failures and disable the webhook.

Flow:
    POST /webhook/github
        ↓
    verify_signature()          → 403 if invalid
        ↓
    parse event + action
        ↓
    trigger_review() (background task)
        ↓
    return 202 Accepted immediately

Why background task:
    GitHub expects a response within 10 seconds. LangGraph reviews
    take 20-60 seconds. We accept immediately and run in the background.
"""

import hashlib
import hmac
import sys
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi import status as http_status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.exceptions import CustomException
from app.core.logger import get_logger
from app.services.review_service import trigger_review

logger = get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

# PR actions that should trigger a review
REVIEW_TRIGGER_ACTIONS = {"opened", "synchronize", "reopened"}


# ── Signature verification ────────────────────────────────────────────────────

def verify_github_signature(
    payload_body: bytes,
    signature_header: Optional[str],
) -> bool:
    """
    Verifies the X-Hub-Signature-256 header sent by GitHub.

    GitHub computes: HMAC-SHA256(secret, raw_body) and sends it as
    "sha256=<hex_digest>". We compute the same and compare using
    hmac.compare_digest() to prevent timing attacks.

    Args:
        payload_body:     Raw request body bytes (must be read before JSON parsing)
        signature_header: Value of X-Hub-Signature-256 header

    Returns:
        True if signature is valid, False otherwise.
    """
    if not signature_header:
        logger.warning("[webhook] Missing X-Hub-Signature-256 header")
        return False

    if not signature_header.startswith("sha256="):
        logger.warning("[webhook] Signature header has unexpected format")
        return False

    secret = settings.github_webhook_secret
    if not secret:
        logger.error("[webhook] GITHUB_WEBHOOK_SECRET not configured")
        return False

    expected_signature = "sha256=" + hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    is_valid = hmac.compare_digest(expected_signature, signature_header)

    if not is_valid:
        logger.warning("[webhook] Signature verification failed")

    return is_valid


# ── Background review task ────────────────────────────────────────────────────

def _run_review_background(
    owner:     str,
    repo:      str,
    pr_number: int,
    db:        Session,
) -> None:
    """
    Runs trigger_review() in a FastAPI BackgroundTask.
    Errors are logged but not re-raised (background tasks cannot
    return errors to the HTTP response).
    """
    logger.info(
        f"[webhook] Background review starting — {owner}/{repo}#{pr_number}"
    )
    try:
        review = trigger_review(owner, repo, pr_number, db)
        logger.info(
            f"[webhook] Background review completed — "
            f"review_id={review.id}, verdict={review.verdict}"
        )
    except Exception as e:
        logger.error(
            f"[webhook] Background review failed — "
            f"{owner}/{repo}#{pr_number}: {e}"
        )
    finally:
        db.close()


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/github",
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Receive GitHub webhook events",
    response_description="Event accepted for processing",
)
async def github_webhook(
    request:          Request,
    background_tasks: BackgroundTasks,
    db:               Session = Depends(get_db),
    x_hub_signature_256: Optional[str] = Header(default=None),
    x_github_event:      Optional[str] = Header(default=None),
) -> dict:
    """
    Receives GitHub webhook POST requests.

    Verifies HMAC signature, then dispatches pull_request events
    to trigger_review() as a background task.

    Returns 202 immediately so GitHub doesn't time out.
    Returns 400 for missing/malformed payloads.
    Returns 403 for invalid signatures.
    Returns 200 with {"status": "ignored"} for non-PR events.
    """
    # ── Step 1: read raw body (must happen before .json()) ───────────────
    raw_body = await request.body()

    logger.info(
        f"[webhook] Received event — "
        f"type={x_github_event}, "
        f"body_size={len(raw_body)} bytes"
    )

    # ── Step 2: verify signature ─────────────────────────────────────────
    if not verify_github_signature(raw_body, x_hub_signature_256):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook signature",
        )

    # ── Step 3: parse payload ─────────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Malformed JSON payload",
        )

    # ── Step 4: ignore non-pull_request events ────────────────────────────
    if x_github_event != "pull_request":
        logger.info(f"[webhook] Ignoring event type: {x_github_event}")
        return {"status": "ignored", "event": x_github_event}

    # ── Step 5: extract PR data ───────────────────────────────────────────
    action     = payload.get("action")
    pr         = payload.get("pull_request", {})
    pr_number  = pr.get("number")
    repository = payload.get("repository", {})
    owner      = repository.get("owner", {}).get("login")
    repo_name  = repository.get("name")

    if not all([action, pr_number, owner, repo_name]):
        logger.warning(
            f"[webhook] Incomplete payload — "
            f"action={action}, pr={pr_number}, owner={owner}, repo={repo_name}"
        )
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Incomplete pull_request payload",
        )

    logger.info(
        f"[webhook] PR event — {owner}/{repo_name}#{pr_number} action={action}"
    )

    # ── Step 6: ignore non-trigger actions ────────────────────────────────
    if action not in REVIEW_TRIGGER_ACTIONS:
        logger.info(f"[webhook] Ignoring PR action: {action}")
        return {"status": "ignored", "action": action}

    # ── Step 7: dispatch review as background task ────────────────────────
    background_tasks.add_task(
        _run_review_background,
        owner=owner,
        repo=repo_name,
        pr_number=pr_number,
        db=db,
    )

    logger.info(
        f"[webhook] Review queued — {owner}/{repo_name}#{pr_number}"
    )

    return {
        "status":    "accepted",
        "message":   f"Review triggered for {owner}/{repo_name}#{pr_number}",
        "pr_number": pr_number,
        "action":    action,
    }