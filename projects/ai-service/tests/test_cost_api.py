from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.java_cost_client import JavaCostClient

OVERVIEW = {
    "by_model": [{"model": "gpt-4o", "call_count": 2, "total_tokens": 450, "estimated_cost": 0.03}],
    "by_intent": [{"intent": "order_query", "call_count": 2, "total_tokens": 450, "estimated_cost": 0.03}],
    "totals": {"call_count": 2, "total_tokens": 450, "estimated_cost": 0.03},
}


def _client(monkeypatch, overview=None, error=None) -> TestClient:
    import app.routers.cost as cost_router

    def fake_fetch_overview(self) -> dict:
        if error is not None:
            raise error
        return overview or OVERVIEW

    monkeypatch.setattr(cost_router.JavaCostClient, "fetch_overview", fake_fetch_overview)
    settings = Settings(_env_file=None)
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_cost_overview_returns_grouped_data(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/ai/cost/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["by_model"][0]["model"] == "gpt-4o"
    assert data["by_intent"][0]["intent"] == "order_query"
    assert data["totals"]["call_count"] == 2


def test_cost_overview_maps_java_error(monkeypatch) -> None:
    from app.services.java_cost_client import JavaCostClientError

    client = _client(monkeypatch, error=JavaCostClientError("java down", status_code=502))
    response = client.get("/api/ai/cost/overview")

    assert response.status_code == 502
    assert response.json()["code"] == "COST_SYNC_FAILED"
