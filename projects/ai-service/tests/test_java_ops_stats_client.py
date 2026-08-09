import httpx
import pytest

from app.services.java_ops_stats_client import JavaOpsStatsClient, JavaOpsStatsClientError

SAMPLE = {
    "feedback": {"helpful": 12, "unhelpful": 3, "helpful_rate": 0.8},
    "handoffs": {"pending": 1, "in_progress": 2, "closed": 5, "total": 8},
    "emotion_distribution": {"angry": 4},
    "conversation_volume": {"2026-08-07": 18},
}


def _client(handler) -> JavaOpsStatsClient:
    transport = httpx.MockTransport(handler)
    return JavaOpsStatsClient(base_url="http://java.test", timeout_seconds=2.0, transport=transport)


def test_get_summary_returns_data():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/ai-ops-stats/summary"
        assert request.url.params["days"] == "7"
        return httpx.Response(
            200,
            json={"success": True, "code": "OK", "message": "ok", "data": SAMPLE, "trace_id": "t"},
        )

    client = _client(handler)
    assert client.get_summary(7) == SAMPLE


def test_get_summary_raises_on_java_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"success": False, "code": "ERR", "message": "boom", "data": None, "trace_id": "t"},
        )

    client = _client(handler)
    with pytest.raises(JavaOpsStatsClientError):
        client.get_summary(7)


def test_get_summary_raises_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = _client(handler)
    with pytest.raises(JavaOpsStatsClientError):
        client.get_summary(7)
