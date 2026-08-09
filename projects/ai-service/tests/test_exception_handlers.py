import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER


def test_not_found_response_is_unified(client: TestClient) -> None:
    response = client.get("/missing", headers={TRACE_ID_HEADER: "trace-not-found"})

    assert response.status_code == 404
    assert response.headers[TRACE_ID_HEADER] == "trace-not-found"
    assert response.json() == {
        "code": "NOT_FOUND",
        "message": "资源不存在",
        "trace_id": "trace-not-found",
    }


def test_validation_error_response_is_unified(client: TestClient) -> None:
    response = client.post("/chat", headers={TRACE_ID_HEADER: "trace-validation"}, json={})
    data = response.json()

    assert response.status_code == 422
    assert response.headers[TRACE_ID_HEADER] == "trace-validation"
    assert data["code"] == "VALIDATION_ERROR"
    assert data["message"] == "请求参数校验失败"
    assert data["trace_id"] == "trace-validation"
    assert data["details"][0]["loc"] == ["body", "message"]
    assert data["details"][0]["type"] == "missing"


def test_app_exception_response_is_unified(app: FastAPI) -> None:
    @app.get("/test/business-error")
    def business_error() -> None:
        raise AppException(
            code="CHAT_DISABLED",
            message="聊天功能暂时不可用",
            status_code=409,
        )

    client = TestClient(app)
    response = client.get(
        "/test/business-error",
        headers={TRACE_ID_HEADER: "trace-business"},
    )

    assert response.status_code == 409
    assert response.headers[TRACE_ID_HEADER] == "trace-business"
    assert response.json() == {
        "code": "CHAT_DISABLED",
        "message": "聊天功能暂时不可用",
        "trace_id": "trace-business",
    }


def test_unhandled_exception_response_is_unified(
    app: FastAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @app.get("/test/unhandled-error")
    def unhandled_error() -> None:
        raise RuntimeError("database is unavailable")

    caplog.set_level(logging.ERROR)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get(
        "/test/unhandled-error",
        headers={TRACE_ID_HEADER: "trace-unhandled"},
    )

    assert response.status_code == 500
    assert response.headers[TRACE_ID_HEADER] == "trace-unhandled"
    assert response.json() == {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "服务器内部错误",
        "trace_id": "trace-unhandled",
    }
    assert "unhandled_exception method=GET path=/test/unhandled-error" in caplog.text
    assert any(
        record.name == "app.core.exception_handlers"
        and record.trace_id == "trace-unhandled"
        for record in caplog.records
    )
