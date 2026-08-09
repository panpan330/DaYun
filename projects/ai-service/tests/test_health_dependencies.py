from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app


def _client(monkeypatch, **probe_results) -> TestClient:
    """Build app with probe functions replaced by scripted results."""
    import app.services.dependency_probe as probe_module

    def _probe(status: str, message: str, latency_ms: float | None) -> tuple:
        return status, message, latency_ms

    probes = {
        "probe_java_business": _probe(*probe_results.get("java", ("ok", "reachable", 1.2))),
        "probe_mcp_product": _probe(*probe_results.get("mcp", ("ok", "reachable", 0.8))),
        "probe_redis": _probe(*probe_results.get("redis", ("ok", "reachable", 0.5))),
        "probe_qdrant": _probe(*probe_results.get("qdrant", ("ok", "reachable", 2.1))),
    }
    for name, result in probes.items():
        monkeypatch.setattr(probe_module, name, lambda settings, r=result: r)

    settings = Settings(_env_file=None)
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_dependencies_check_returns_ok_when_all_reachable(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/health/dependencies")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["service"] == "ai-service"
    names = [dep["name"] for dep in data["dependencies"]]
    assert names == ["java_business", "mcp_product", "redis", "qdrant"]
    assert all(dep["status"] == "ok" for dep in data["dependencies"])
    assert all(isinstance(dep["latency_ms"], float) for dep in data["dependencies"])


def test_dependencies_check_reports_degraded_when_java_down(monkeypatch) -> None:
    client = _client(monkeypatch, java=("unreachable", "connection refused", None))
    response = client.get("/health/dependencies")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "degraded"
    by_name = {dep["name"]: dep for dep in data["dependencies"]}
    assert by_name["java_business"]["status"] == "unreachable"
    assert by_name["mcp_product"]["status"] == "ok"


def test_dependencies_check_reports_degraded_when_mcp_down(monkeypatch) -> None:
    client = _client(monkeypatch, mcp=("unreachable", "connection refused", None))
    response = client.get("/health/dependencies")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "degraded"
    by_name = {dep["name"]: dep for dep in data["dependencies"]}
    assert by_name["mcp_product"]["status"] == "unreachable"


def test_dependencies_check_reports_not_configured_for_missing_qdrant(monkeypatch) -> None:
    client = _client(monkeypatch, qdrant=("not_configured", "Qdrant base URL is empty.", None))
    response = client.get("/health/dependencies")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "degraded"
    by_name = {dep["name"]: dep for dep in data["dependencies"]}
    assert by_name["qdrant"]["status"] == "not_configured"


def test_dependencies_check_all_unreachable_still_returns_200(monkeypatch) -> None:
    client = _client(
        monkeypatch,
        java=("unreachable", "down", None),
        mcp=("unreachable", "down", None),
        redis=("unreachable", "down", None),
        qdrant=("unreachable", "down", None),
    )
    response = client.get("/health/dependencies")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
