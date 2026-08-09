import json

import httpx
import pytest

from app.core.business_context import reset_business_context, set_business_context
from app.core.config import Settings
from app.core.trace import TRACE_ID_HEADER, reset_trace_id, set_trace_id
from app.services.java_conversation_client import (
    JavaConversationClient,
    JavaConversationClientError,
)


@pytest.fixture(autouse=True)
def _business_and_trace_context() -> None:
    tokens = set_business_context(user_id="U1001", tenant_id="default")
    trace_token = set_trace_id("trace-conv-001")
    yield
    reset_business_context(tokens)
    reset_trace_id(trace_token)


def make_client(handler) -> JavaConversationClient:
    settings = Settings()
    return JavaConversationClient(
        base_url="http://java.test",
        timeout_seconds=2.0,
        settings=settings,
        transport=httpx.MockTransport(handler),
    )


def capture(handler):
    received: dict[str, object] = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        received["method"] = request.method
        received["path"] = request.url.path
        received["query"] = request.url.params
        received["body"] = json.loads(request.content.decode("utf-8")) if request.content else None
        received["internal_token"] = request.headers.get("X-Internal-Token")
        received["user_id"] = request.headers.get("X-User-Id")
        received["tenant_id"] = request.headers.get("X-Tenant-Id")
        received["trace_id"] = request.headers.get(TRACE_ID_HEADER)
        return handler(request)

    return received, wrapped


def test_upsert_conversation_sends_valid_payload():
    received, wrapped = capture(
        lambda request: httpx.Response(200, json={"success": True, "data": None, "trace_id": "t"})
    )
    client = make_client(wrapped)

    client.upsert_conversation(
        tenant_id="default",
        conversation_id="conv-1",
        user_id="U1001",
        title="我的订单",
        conversation_status="active",
    )

    assert received["method"] == "POST"
    assert received["path"] == "/internal/ai-conversations"
    assert received["body"] == {
        "conversation_id": "conv-1",
        "user_id": "U1001",
        "title": "我的订单",
        "conversation_status": "active",
    }
    assert received["internal_token"] == "local-dev-internal-token"
    assert received["user_id"] == "U1001"
    assert received["tenant_id"] == "default"
    assert received["trace_id"] == "trace-conv-001"


def test_batch_write_messages_returns_inserted_count():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/ai-conversations/messages"
        return httpx.Response(200, json={"success": True, "data": {"inserted": 2}, "trace_id": "t"})

    client = make_client(handler)

    inserted = client.batch_write_messages(
        tenant_id="default",
        conversation_id="conv-1",
        messages=[
            {"message_id": "m-1", "sender_type": "user", "content": "你好", "trace_id": "tr-1"},
            {"message_id": "m-2", "sender_type": "assistant", "content": "您好", "trace_id": "tr-2"},
        ],
    )

    assert inserted == 2


def test_list_conversations_sends_query_and_parses_data():
    received, wrapped = capture(
        lambda request: httpx.Response(
            200,
            json={
                "success": True,
                "data": [
                    {
                        "conversation_id": "conv-2",
                        "user_id": "U1001",
                        "title": "title-2",
                        "conversation_status": "active",
                        "updated_at": "2026-08-08T10:00:00Z",
                    },
                    {
                        "conversation_id": "conv-1",
                        "user_id": "U1001",
                        "title": "title-1",
                        "conversation_status": "completed",
                        "updated_at": "2026-08-07T10:00:00Z",
                    },
                ],
                "trace_id": "t",
            },
        )
    )
    client = make_client(wrapped)

    result = client.list_conversations(tenant_id="default", user_id="U1001", limit=20)

    assert received["path"] == "/internal/ai-conversations"
    assert str(received["query"]) == "limit=20"
    assert len(result) == 2
    assert result[0].conversation_id == "conv-2"
    assert result[0].conversation_status == "active"
    assert result[1].title == "title-1"


def test_get_messages_parses_ascending_list():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/ai-conversations/conv-1/messages"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": [
                    {
                        "message_id": "m-1",
                        "conversation_id": "conv-1",
                        "sender_type": "user",
                        "content": "你好",
                        "trace_id": "tr-1",
                        "created_at": "2026-08-08T09:00:00Z",
                    },
                    {
                        "message_id": "m-2",
                        "conversation_id": "conv-1",
                        "sender_type": "assistant",
                        "content": "您好",
                        "trace_id": "tr-2",
                        "created_at": "2026-08-08T09:01:00Z",
                    },
                ],
                "trace_id": "t",
            },
        )

    client = make_client(handler)

    result = client.get_messages(tenant_id="default", conversation_id="conv-1")

    assert len(result) == 2
    assert result[0].message_id == "m-1"
    assert result[0].sender_type == "user"
    assert result[1].sender_type == "assistant"


def test_server_error_raises_client_error():
    client = make_client(lambda request: httpx.Response(500, json={"success": False}))

    with pytest.raises(JavaConversationClientError) as exc_info:
        client.list_conversations(tenant_id="default", user_id="U1001", limit=20)

    assert exc_info.value.status_code == 500


def test_connection_error_raises_client_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)

    with pytest.raises(JavaConversationClientError):
        client.batch_write_messages(
            tenant_id="default",
            conversation_id="conv-1",
            messages=[{"message_id": "m-1", "sender_type": "user", "content": "x", "trace_id": "t"}],
        )


def test_background_call_always_sends_trace_id_header():
    """Background tasks (no request trace context) must still send X-Trace-Id,
    otherwise Java internal auth rejects the call with 401."""
    from app.core.trace import reset_trace_id, set_trace_id

    token = set_trace_id("-")  # default placeholder = no trace context
    try:
        received, wrapped = capture(
            lambda request: httpx.Response(200, json={"success": True, "data": None, "trace_id": "t"})
        )
        client = make_client(wrapped)
        client.upsert_conversation(
            tenant_id="default",
            conversation_id="conv-1",
            user_id="U1001",
            title="t",
            conversation_status="active",
        )
    finally:
        reset_trace_id(token)

    trace_id = received["trace_id"]
    assert isinstance(trace_id, str)
    assert len(trace_id) >= 8
    assert trace_id.startswith("sync-")
