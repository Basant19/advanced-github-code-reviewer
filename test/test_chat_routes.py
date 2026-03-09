"""
tests/test_chat_routes.py

Unit Tests for Chat API Routes
--------------------------------
Tests all three chat endpoints using FastAPI TestClient.
ChatService is fully mocked — no DB calls.

Run with:
    pytest tests/test_chat_routes.py -v

What is mocked:
    - ChatService         → prevents real DB access
    - get_db()            → yields a MagicMock session
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.api.routes.chat import router
from app.api.deps import get_db
from app.core.exceptions import CustomException

# ──────────────────────────────────────────────────────────────────────────────
# Test app setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(router)


def override_get_db():
    yield MagicMock()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

NOW = datetime.now(timezone.utc)
THREAD_ID = "Basant19-my-repo-7"


def make_mock_message(
    id:        int = 1,
    thread_id: int = 1,
    role:      str = "user",
    content:   str = "What does this change do?",
) -> MagicMock:
    m            = MagicMock()
    m.id         = id
    m.thread_id  = thread_id
    m.role       = role
    m.content    = content
    m.created_at = NOW
    return m


def make_mock_chat_service(
    messages:       list = None,
    created_message: MagicMock = None,
) -> MagicMock:
    svc = MagicMock()
    svc.create_message.return_value = created_message or make_mock_message()
    svc.get_messages.return_value   = messages or []
    svc.delete_thread.return_value  = None
    return svc


# ──────────────────────────────────────────────────────────────────────────────
# Test: POST /chat/{thread_id}/messages
# ──────────────────────────────────────────────────────────────────────────────

class TestSendMessage:

    def test_returns_201_on_success(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "What does this change do?"},
            )

        assert response.status_code == 201

    def test_response_contains_message_fields(self):
        mock_msg = make_mock_message(id=5, role="user",
                                      content="Explain this diff")
        mock_svc = make_mock_chat_service(created_message=mock_msg)

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Explain this diff"},
            )

        data = response.json()
        assert data["id"]      == 5
        assert data["role"]    == "user"
        assert data["content"] == "Explain this diff"

    def test_default_role_is_user(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Hello"},   # no role specified
            )

        call_kwargs = mock_svc.create_message.call_args[1]
        assert call_kwargs["role"] == "user"

    def test_custom_role_is_forwarded(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Context", "role": "system"},
            )

        call_kwargs = mock_svc.create_message.call_args[1]
        assert call_kwargs["role"] == "system"

    def test_thread_id_is_forwarded_to_service(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Hello"},
            )

        call_kwargs = mock_svc.create_message.call_args[1]
        assert call_kwargs["thread_id"] == THREAD_ID

    def test_content_is_forwarded_to_service(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Is this safe to merge?"},
            )

        call_kwargs = mock_svc.create_message.call_args[1]
        assert call_kwargs["content"] == "Is this safe to merge?"

    def test_returns_500_on_service_error(self):
        mock_svc = MagicMock()
        mock_svc.create_message.side_effect = CustomException(
            "DB write failed", sys
        )

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.post(
                f"/chat/{THREAD_ID}/messages",
                json={"content": "Hello"},
            )

        assert response.status_code == 500

    def test_returns_422_when_content_missing(self):
        response = client.post(
            f"/chat/{THREAD_ID}/messages",
            json={},   # missing required content field
        )
        assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Test: GET /chat/{thread_id}/messages
# ──────────────────────────────────────────────────────────────────────────────

class TestGetMessages:

    def test_returns_200_on_success(self):
        mock_svc = make_mock_chat_service(messages=[])

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        assert response.status_code == 200

    def test_response_contains_thread_id(self):
        mock_svc = make_mock_chat_service(messages=[])

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        assert response.json()["thread_id"] == THREAD_ID

    def test_response_contains_message_count(self):
        messages = [make_mock_message(id=i) for i in range(3)]
        mock_svc = make_mock_chat_service(messages=messages)

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        assert response.json()["message_count"] == 3

    def test_response_contains_messages_list(self):
        messages = [
            make_mock_message(id=1, role="user",  content="Hello"),
            make_mock_message(id=2, role="ai",    content="Hi there"),
        ]
        mock_svc = make_mock_chat_service(messages=messages)

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        data = response.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "ai"

    def test_empty_thread_returns_empty_list(self):
        mock_svc = make_mock_chat_service(messages=[])

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        data = response.json()
        assert data["messages"]      == []
        assert data["message_count"] == 0

    def test_thread_id_forwarded_to_service(self):
        mock_svc = make_mock_chat_service(messages=[])

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.get(f"/chat/{THREAD_ID}/messages")

        call_kwargs = mock_svc.get_messages.call_args[1]
        assert call_kwargs["thread_id"] == THREAD_ID

    def test_returns_404_when_thread_not_found(self):
        mock_svc = MagicMock()
        mock_svc.get_messages.side_effect = CustomException(
            "Thread not found", sys
        )

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get("/chat/nonexistent-thread/messages")

        assert response.status_code == 404

    def test_returns_500_on_service_error(self):
        mock_svc = MagicMock()
        mock_svc.get_messages.side_effect = CustomException(
            "DB read failed", sys
        )

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.get(f"/chat/{THREAD_ID}/messages")

        assert response.status_code == 500


# ──────────────────────────────────────────────────────────────────────────────
# Test: DELETE /chat/{thread_id}
# ──────────────────────────────────────────────────────────────────────────────

class TestDeleteThread:

    def test_returns_200_on_success(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete(f"/chat/{THREAD_ID}")

        assert response.status_code == 200

    def test_response_contains_deleted_true(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete(f"/chat/{THREAD_ID}")

        assert response.json()["deleted"] is True

    def test_response_contains_thread_id(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete(f"/chat/{THREAD_ID}")

        assert response.json()["thread_id"] == THREAD_ID

    def test_response_contains_message(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete(f"/chat/{THREAD_ID}")

        assert "message" in response.json()

    def test_thread_id_forwarded_to_service(self):
        mock_svc = make_mock_chat_service()

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            client.delete(f"/chat/{THREAD_ID}")

        call_kwargs = mock_svc.delete_thread.call_args[1]
        assert call_kwargs["thread_id"] == THREAD_ID

    def test_returns_404_when_thread_not_found(self):
        mock_svc = MagicMock()
        mock_svc.delete_thread.side_effect = CustomException(
            "Thread not found", sys
        )

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete("/chat/nonexistent-thread")

        assert response.status_code == 404

    def test_returns_500_on_service_error(self):
        mock_svc = MagicMock()
        mock_svc.delete_thread.side_effect = CustomException(
            "DB delete failed", sys
        )

        with patch("app.api.routes.chat.ChatService", return_value=mock_svc):
            response = client.delete(f"/chat/{THREAD_ID}")

        assert response.status_code == 500