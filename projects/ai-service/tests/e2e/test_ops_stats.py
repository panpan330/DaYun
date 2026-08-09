import httpx
import pytest

pytestmark = pytest.mark.e2e


def test_ops_stats_summary_returns_four_sections() -> None:
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as client:
        response = client.get("/api/ai/ops-stats/summary?days=7")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) >= {"feedback", "handoffs", "emotion_distribution", "conversation_volume"}
        assert isinstance(body["feedback"]["helpful"], int)
        assert isinstance(body["handoffs"]["total"], int)
        assert isinstance(body["emotion_distribution"], dict)
        assert isinstance(body["conversation_volume"], dict)
