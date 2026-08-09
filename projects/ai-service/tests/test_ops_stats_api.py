from fastapi.testclient import TestClient

from app.main import app
from app.routers.ops_stats import get_ops_stats_client
from app.services.java_ops_stats_client import JavaOpsStatsClientError

SAMPLE = {
    "feedback": {"helpful": 1, "unhelpful": 0, "helpful_rate": 1.0},
    "handoffs": {"pending": 0, "in_progress": 0, "closed": 0, "total": 0},
    "emotion_distribution": {},
    "conversation_volume": {},
}


def _install_fake_client(monkeypatch, impl) -> None:
    class FakeClient:
        def get_summary(self, days: int):
            return impl(days)

    # monkeypatch 在测试结束时自动还原 dependency_overrides
    monkeypatch.setitem(app.dependency_overrides, get_ops_stats_client, lambda: FakeClient())


def test_ops_stats_summary_ok(monkeypatch):
    _install_fake_client(monkeypatch, lambda days: SAMPLE)
    client = TestClient(app)
    response = client.get("/api/ai/ops-stats/summary?days=7")
    assert response.status_code == 200
    assert response.json()["feedback"]["helpful"] == 1


def test_ops_stats_summary_rejects_invalid_days():
    client = TestClient(app)
    response = client.get("/api/ai/ops-stats/summary?days=5")
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_DAYS"


def test_ops_stats_summary_java_down(monkeypatch):
    def boom(_days):
        raise JavaOpsStatsClientError("java down", status_code=502, code="OPS_STATS_UNAVAILABLE")

    _install_fake_client(monkeypatch, boom)
    client = TestClient(app)
    response = client.get("/api/ai/ops-stats/summary?days=7")
    assert response.status_code == 502
    assert response.json()["code"] == "OPS_STATS_UNAVAILABLE"
