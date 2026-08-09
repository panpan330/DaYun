import httpx
import pytest

from app.services.java_cost_client import JavaCostClient, JavaCostClientError

SAMPLE_RECORD = {
    "model": "gpt-4o",
    "intent": "general",
    "call_count": 1,
    "input_tokens": 1,
    "output_tokens": 1,
    "total_tokens": 2,
    "estimated_cost": 0.001,
    "window_start": "2026-08-08T00:00:00Z",
    "window_end": "2026-08-08T00:00:00Z",
}


def _client(handler) -> JavaCostClient:
    transport = httpx.MockTransport(handler)
    return JavaCostClient(base_url="http://java.test", timeout_seconds=2.0, transport=transport)


def test_sync_cost_records_posts_batch():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    client.sync_cost_records([SAMPLE_RECORD])
    assert seen["path"] == "/internal/ai-cost-records"
    assert b"gpt-4o" in seen["body"]


def test_fetch_overview_returns_grouped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": True,
                "code": "OK",
                "message": "ok",
                "data": {
                    "by_model": [{"model": "gpt-4o", "call_count": 1}],
                    "by_intent": [],
                    "totals": {"call_count": 1},
                },
                "trace_id": "t",
            },
        )

    client = _client(handler)
    overview = client.fetch_overview()
    assert overview["by_model"][0]["model"] == "gpt-4o"
    assert overview["totals"]["call_count"] == 1


def test_sync_raises_on_non_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            json={
                "success": False,
                "code": "MCP_SERVER_UNREACHABLE",
                "message": "down",
                "data": None,
                "trace_id": "t",
            },
        )

    client = _client(handler)
    with pytest.raises(JavaCostClientError):
        client.sync_cost_records([SAMPLE_RECORD])


def test_sync_raises_on_request_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    with pytest.raises(JavaCostClientError):
        client.sync_cost_records([SAMPLE_RECORD])


def test_sync_sends_trace_id_without_request_context():
    # 后台线程无请求 trace 上下文时，须带 sync- 兜底 X-Trace-Id（Java internal 鉴权要求）
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["trace_id"] = request.headers.get("X-Trace-Id")
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t"},
        )

    from app.core.trace import TRACE_ID_HEADER, DEFAULT_TRACE_ID
    from app.core.trace import _trace_id

    token = _trace_id.set(DEFAULT_TRACE_ID)
    try:
        client = _client(handler)
        client.sync_cost_records([SAMPLE_RECORD])
    finally:
        _trace_id.reset(token)
    assert seen["trace_id"] is not None
    assert seen["trace_id"].startswith("sync-")
    assert len(seen["trace_id"]) >= 8
