import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.trace import (
    DEFAULT_TRACE_ID,
    TRACE_ID_HEADER,
    build_trace_headers,
    generate_trace_id,
    get_or_create_trace_id,
    get_trace_id,
    reset_trace_id,
    set_trace_id,
)
from app.routers.chat import get_llm_chat_service
from app.schemas.chat import ChatMessage


class FakeLLMChatService:
    def generate_reply(
        self,
        user_message: str,
        *,
        history: list[ChatMessage] | None = None,
    ) -> str:
        return f"测试回复：{user_message}"


def test_generate_trace_id_returns_hex_string() -> None:
    trace_id = generate_trace_id()

    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)


def test_get_or_create_trace_id_reuses_incoming_header() -> None:
    assert get_or_create_trace_id("client-trace-001") == "client-trace-001"


def test_get_or_create_trace_id_ignores_blank_header() -> None:
    trace_id = get_or_create_trace_id("   ")

    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)


def test_trace_id_context_can_be_set_and_reset() -> None:
    token = set_trace_id("lesson-13")

    try:
        assert get_trace_id() == "lesson-13"
    finally:
        reset_trace_id(token)

    assert get_trace_id() == DEFAULT_TRACE_ID


def test_build_trace_headers_uses_current_trace_id_only_when_available() -> None:
    assert build_trace_headers() == {}

    token = set_trace_id("trace-outgoing-001")
    try:
        assert build_trace_headers() == {TRACE_ID_HEADER: "trace-outgoing-001"}
    finally:
        reset_trace_id(token)


def test_health_response_has_trace_id_header(client: TestClient) -> None:
    response = client.get("/health")
    trace_id = response.headers[TRACE_ID_HEADER]

    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", trace_id)


def test_trace_id_header_reuses_incoming_value(client: TestClient) -> None:
    response = client.get("/health", headers={TRACE_ID_HEADER: "client-trace-001"})

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "client-trace-001"


def test_trace_id_header_is_different_for_different_requests(
    client: TestClient,
) -> None:
    first = client.get("/health")
    second = client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers[TRACE_ID_HEADER] != second.headers[TRACE_ID_HEADER]


def test_chat_logs_share_request_trace_id(
    app: FastAPI,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trace_id = "client-trace-lesson-13"
    app.dependency_overrides[get_llm_chat_service] = lambda: FakeLLMChatService()
    caplog.set_level(logging.INFO)

    response = client.post(
        "/chat",
        headers={TRACE_ID_HEADER: trace_id},
        json={"message": "追踪日志"},
    )

    messages = [record.getMessage() for record in caplog.records]
    trace_ids = [
        record.trace_id
        for record in caplog.records
        if record.name in {"app.middleware.tracing", "app.routers.chat"}
    ]

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == trace_id
    assert "request_started method=POST path=/chat" in messages
    assert "chat_requested message_length=4 history_size=0" in messages
    assert any(message.startswith("request_finished method=POST path=/chat") for message in messages)
    assert trace_ids
    assert all(value == trace_id for value in trace_ids)
