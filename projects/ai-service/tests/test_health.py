from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["service"] == "ai-service"
    assert isinstance(data["time"], str)


def test_readiness_check_returns_ready_for_rule_based_mode(
    client: TestClient,
) -> None:
    response = client.get("/ready")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ready"
    assert data["service"] == "ai-service"
    assert data["ready"] is True
    assert _check_status(data, "java_business_service_base_url") == "configured"
    assert _check_status(data, "llm_api_key") == "skipped"


def test_readiness_check_requires_api_key_for_real_llm_mode() -> None:
    settings = Settings(
        ticket_agent_model_mode="real_llm",
        llm_api_key=None,
        openai_api_key=None,
        _env_file=None,
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    response = client.get("/ready")
    data = response.json()

    assert response.status_code == 503
    assert data["status"] == "not_ready"
    assert data["ready"] is False
    assert _check_status(data, "llm_api_key") == "not_configured"


def test_readiness_check_accepts_real_llm_mode_with_api_key() -> None:
    settings = Settings(
        ticket_agent_model_mode="real_llm",
        llm_api_key="test-key",
        _env_file=None,
    )
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app)

    response = client.get("/ready")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "ready"
    assert data["ready"] is True
    assert _check_status(data, "llm_api_key") == "configured"


def _check_status(data: dict[str, object], name: str) -> str:
    checks = data["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check["name"] == name:
            status = check["status"]
            assert isinstance(status, str)
            return status
    raise AssertionError(f"missing readiness check: {name}")
