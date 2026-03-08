import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.chat_service import ChatService
from schemas.chat_schema import MessageCreate


@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def chat_service(mock_db):
    return ChatService(mock_db)


@pytest.mark.asyncio
async def test_create_message(chat_service, mock_db):

    message_data = MessageCreate(
        thread_id=1,
        role="user",
        content="Why did the AI request changes?"
    )

    mock_message = MagicMock()
    mock_message.id = 10
    mock_message.thread_id = 1
    mock_message.role = "user"
    mock_message.content = "Why did the AI request changes?"

    chat_service.db.add = MagicMock()
    chat_service.db.commit = AsyncMock()
    chat_service.db.refresh = AsyncMock()

    async def refresh_side_effect(msg):
        msg.id = 10

    chat_service.db.refresh.side_effect = refresh_side_effect

    result = await chat_service.create_message(message_data)

    assert result.thread_id == 1
    assert result.role == "user"
    assert result.message_id == 10


@pytest.mark.asyncio
async def test_get_thread_messages(chat_service, mock_db):

    mock_message = MagicMock()
    mock_message.thread_id = 1
    mock_message.id = 1
    mock_message.role = "user"
    mock_message.content = "Test message"

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_message]

    mock_db.execute.return_value = mock_result

    messages = await chat_service.get_thread_messages(1)

    assert len(messages) == 1
    assert messages[0].content == "Test message"


@pytest.mark.asyncio
async def test_delete_message(chat_service, mock_db):

    mock_message = MagicMock()
    mock_message.id = 5

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_message

    mock_db.execute.return_value = mock_result
    mock_db.delete = AsyncMock()
    mock_db.commit = AsyncMock()

    result = await chat_service.delete_message(5)

    assert result is True