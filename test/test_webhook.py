"""
tests/test_webhook.py

Unit Tests for GitHub Webhook Route
--------------------------------------
Tests signature verification, event routing, and background task dispatch
using FastAPI TestClient — no real GitHub requests or DB calls.

Run with:
    pytest tests/test_webhook.py -v

What is mocked:
    - trigger_review()          → prevents real LangGraph execution
    - get_db()                  → yields a MagicMock session
    - settings.github_webhook_secret → controlled test secret
"""

import hashlib
import hmac
import json
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.routes.webhook import router, verify_github_signature
from app.api.deps import get_db

# ──────────────────────────────────────────────────────────────────────────────
# Test app setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(router)

TEST_SECRET = "test_webhook_secret_12345"


def make_mock_db():
    db = MagicMock()
    db.close = MagicMock()
    return db


def override_get_db():
    yield make_mock_db()


app.dependency_overrides[get_db] = override_get_db


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_signature(body: bytes, secret: str = TEST_SECRET) -> str:
    """Computes a valid X-Hub-Signature-256 for the given body."""
    return "sha256=" + hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()


def make_pr_payload(
    action: str    = "opened",
    pr_number: int = 7,
    owner: str     = "Basant19",
    repo: str      = "advanced-github-code-reviewer",
) -> dict:
    return {
        "action": action,
        "pull_request": {
            "number": pr_number,
            "title":  "Add review service",
            "state":  "open",
        },
        "repository": {
            "name":  repo,
            "owner": {"login": owner},
        },
    }


def post_webhook(
    client:  TestClient,
    payload: dict,
    event:   str  = "pull_request",
    secret:  str  = TEST_SECRET,
    sign:    bool = True,
) -> object:
    """Helper to POST a webhook event with correct headers."""
    body    = json.dumps(payload).encode("utf-8")
    headers = {"X-GitHub-Event": event}

    if sign:
        headers["X-Hub-Signature-256"] = make_signature(body, secret)

    return client.post("/webhook/github", content=body, headers=headers)


# ──────────────────────────────────────────────────────────────────────────────
# Test: verify_github_signature  (pure function)
# ──────────────────────────────────────────────────────────────────────────────

class TestVerifyGithubSignature:

    def test_valid_signature_returns_true(self):
        from app.api.routes.webhook import verify_github_signature
        body      = b'{"action": "opened"}'
        signature = make_signature(body)

        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            result = verify_github_signature(body, signature)

        assert result is True

    def test_wrong_secret_returns_false(self):
        from app.api.routes.webhook import verify_github_signature
        body      = b'{"action": "opened"}'
        signature = make_signature(body, secret="wrong_secret")

        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            result = verify_github_signature(body, signature)

        assert result is False

    def test_tampered_body_returns_false(self):
        from app.api.routes.webhook import verify_github_signature
        body           = b'{"action": "opened"}'
        tampered_body  = b'{"action": "deleted"}'
        signature      = make_signature(body)

        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            result = verify_github_signature(tampered_body, signature)

        assert result is False

    def test_missing_signature_header_returns_false(self):
        from app.api.routes.webhook import verify_github_signature
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            result = verify_github_signature(b"body", None)

        assert result is False

    def test_signature_without_sha256_prefix_returns_false(self):
        from app.api.routes.webhook import verify_github_signature
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            result = verify_github_signature(b"body", "abc123")

        assert result is False

    def test_missing_webhook_secret_config_returns_false(self):
        from app.api.routes.webhook import verify_github_signature
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = ""
            result = verify_github_signature(b"body", "sha256=abc")

        assert result is False


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /webhook/github — signature enforcement
# ──────────────────────────────────────────────────────────────────────────────

class TestWebhookSignatureEnforcement:

    def test_missing_signature_returns_403(self):
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            payload  = make_pr_payload()
            body     = json.dumps(payload).encode()
            response = client.post(
                "/webhook/github",
                content=body,
                headers={"X-GitHub-Event": "pull_request"},
            )

        assert response.status_code == 403

    def test_invalid_signature_returns_403(self):
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client  = TestClient(app)
            payload = make_pr_payload()
            body    = json.dumps(payload).encode()
            response = client.post(
                "/webhook/github",
                content=body,
                headers={
                    "X-GitHub-Event":        "pull_request",
                    "X-Hub-Signature-256":   "sha256=deadbeef",
                },
            )

        assert response.status_code == 403

    def test_valid_signature_does_not_return_403(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook.trigger_review") as mock_trigger:
            mock_settings.github_webhook_secret = TEST_SECRET
            mock_trigger.return_value = MagicMock(id=1, verdict="APPROVE")

            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload())

        assert response.status_code != 403


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /webhook/github — event routing
# ──────────────────────────────────────────────────────────────────────────────

class TestWebhookEventRouting:

    def test_non_pull_request_event_returns_ignored(self):
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(
                client,
                payload={"ref": "refs/heads/main"},
                event="push",
            )

        # FastAPI applies the decorator's status_code=202 to all responses
        # from this endpoint, including ignored events. Body still signals ignored.
        assert response.status_code == 202
        assert response.json()["status"] == "ignored"

    def test_ignored_pr_action_returns_ignored(self):
        """Actions like 'closed', 'labeled' should not trigger a review."""
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(action="closed"))

        # Same reason — 202 is the endpoint default, body still signals ignored.
        assert response.status_code == 202
        assert response.json()["status"] == "ignored"

    def test_opened_action_triggers_review(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background") as mock_bg:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(action="opened"))

        assert response.status_code == 202
        assert response.json()["status"] == "accepted"

    def test_synchronize_action_triggers_review(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background"):
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(action="synchronize"))

        assert response.status_code == 202

    def test_reopened_action_triggers_review(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background"):
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(action="reopened"))

        assert response.status_code == 202


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /webhook/github — response payload
# ──────────────────────────────────────────────────────────────────────────────

class TestWebhookResponsePayload:

    def test_accepted_response_contains_pr_number(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background"):
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(pr_number=42))

        data = response.json()
        assert data["pr_number"] == 42

    def test_accepted_response_contains_action(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background"):
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload(action="opened"))

        assert response.json()["action"] == "opened"

    def test_accepted_response_contains_message(self):
        with patch("app.api.routes.webhook.settings") as mock_settings, \
             patch("app.api.routes.webhook._run_review_background"):
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, make_pr_payload())

        assert "message" in response.json()


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /webhook/github — malformed payloads
# ──────────────────────────────────────────────────────────────────────────────

class TestWebhookMalformedPayloads:

    def test_missing_pr_number_returns_400(self):
        payload = {
            "action": "opened",
            "pull_request": {},          # no number
            "repository": {
                "name":  "repo",
                "owner": {"login": "owner"},
            },
        }
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, payload)

        assert response.status_code == 400

    def test_missing_repository_owner_returns_400(self):
        payload = {
            "action": "opened",
            "pull_request": {"number": 7},
            "repository": {"name": "repo"},   # no owner
        }
        with patch("app.api.routes.webhook.settings") as mock_settings:
            mock_settings.github_webhook_secret = TEST_SECRET
            client   = TestClient(app)
            response = post_webhook(client, payload)

        assert response.status_code == 400