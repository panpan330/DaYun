import httpx
import pytest

pytestmark = pytest.mark.e2e


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer local-dev-token:default:{user_id}"}


def test_services_healthy() -> None:
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_customer_token_authenticated() -> None:
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as client:
        response = client.get(
            "/api/ai/agent/conversations?limit=1",
            headers=_auth("U1001"),
        )
        assert response.status_code == 200
