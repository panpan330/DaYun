from fastapi.testclient import TestClient


def test_allowed_origin_gets_cors_header(client: TestClient) -> None:
    origin = "http://localhost:5173"

    response = client.get("/health", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_local_vite_port_gets_cors_header(client: TestClient) -> None:
    origin = "http://localhost:5174"

    response = client.options(
        "/api/ai/evaluation/overview",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_disallowed_origin_does_not_get_cors_header(client: TestClient) -> None:
    response = client.get(
        "/health",
        headers={"Origin": "https://evil.example"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_preflight_request_allows_configured_origin(client: TestClient) -> None:
    origin = "http://localhost:5173"

    response = client.options(
        "/chat",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]


def test_preflight_request_allows_ticket_confirmation_correction(client: TestClient) -> None:
    origin = "http://localhost:5173"

    response = client.options(
        "/api/ai/agent/conversations/example/confirmations/example",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "PUT" in response.headers["access-control-allow-methods"]


def test_preflight_request_rejects_disallowed_origin(client: TestClient) -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
