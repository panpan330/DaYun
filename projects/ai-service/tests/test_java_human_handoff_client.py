import httpx
import pytest

from app.services.java_human_handoff_client import (
    JavaHumanHandoffClient,
    JavaHumanHandoffClientError,
)

SAMPLE = {
    "conversation_id": "conv-1",
    "user_id": "U1001",
    "tenant_id": "default",
    "reason": "检测到强烈情绪（angry），建议由人工客服继续跟进。",
    "related_order_id": "202501010001",
    "emotion": "angry",
}


def _client(handler) -> JavaHumanHandoffClient:
    transport = httpx.MockTransport(handler)
    return JavaHumanHandoffClient(base_url="http://java.test", timeout_seconds=2.0, transport=transport)


def test_create_handoff_posts_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    client.create_handoff(**SAMPLE)
    assert seen["path"] == "/internal/ai-human-handoffs"
    assert b"conv-1" in seen["body"]
    assert b"angry" in seen["body"]


def test_create_handoff_accepts_success_when_active_exists():
    # 幂等由 Java 侧保证（active 记录存在时跳过），client 只需接受 success
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    client.create_handoff(**SAMPLE)


def test_create_handoff_raises_on_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"success": False, "code": "INTERNAL_AUTH_FAILED", "message": "auth", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    with pytest.raises(JavaHumanHandoffClientError):
        client.create_handoff(**SAMPLE)


def test_create_handoff_raises_on_request_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    with pytest.raises(JavaHumanHandoffClientError):
        client.create_handoff(**SAMPLE)


def test_create_handoff_sends_trace_id_without_request_context():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["trace_id"] = request.headers.get("X-Trace-Id")
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    from app.core.trace import DEFAULT_TRACE_ID, _trace_id

    token = _trace_id.set(DEFAULT_TRACE_ID)
    try:
        client = _client(handler)
        client.create_handoff(**SAMPLE)
    finally:
        _trace_id.reset(token)
    assert seen["trace_id"] is not None
    assert seen["trace_id"].startswith("sync-")
    assert len(seen["trace_id"]) >= 8


def test_get_active_handoff_returns_handoff():
    payload = {
        "success": True,
        "code": "OK",
        "message": "ok",
        "trace_id": "t-1",
        "data": {
            "id": 1,
            "conversation_id": "conv-1",
            "status": "in_progress",
            "assigned_agent": "A1001",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/internal/ai-human-handoffs/active-by-conversation"
        assert request.url.params["conversationId"] == "conv-1"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    result = client.get_active_handoff("conv-1")
    assert result == payload["data"]


def test_get_active_handoff_returns_none_when_no_active():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    assert client.get_active_handoff("conv-1") is None


def test_get_active_handoff_raises_on_java_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"success": False, "code": "INTERNAL_ERROR", "message": "boom", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    with pytest.raises(JavaHumanHandoffClientError):
        client.get_active_handoff("conv-1")
