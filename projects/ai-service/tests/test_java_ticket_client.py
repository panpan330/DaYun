from datetime import datetime, timezone
import json
import logging

import httpx
import pytest

from app.core.business_context import reset_business_context, set_business_context
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER, reset_trace_id, set_trace_id
from app.schemas.ticket import CreateTicketArgs
from app.services.java_ticket_client import JavaTicketClient


def make_arguments() -> CreateTicketArgs:
    return CreateTicketArgs(
        requester_id="demo_user_001",
        title="订单 A1001 一直未发货",
        description="订单 A1001 已付款一周仍未发货，请帮我处理。",
        category="complaint",
        priority="high",
        related_order_id="A1001",
    )


def test_java_ticket_client_sends_validated_arguments_and_validates_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    received_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received_request["method"] = request.method
        received_request["path"] = request.url.path
        received_request["body"] = json.loads(request.content.decode("utf-8"))
        received_request["idempotency_key"] = request.headers["Idempotency-Key"]
        received_request["trace_id"] = request.headers[TRACE_ID_HEADER]
        received_request["caller"] = request.headers["X-Caller"]
        received_request["user_id"] = request.headers["X-User-Id"]
        received_request["tenant_id"] = request.headers["X-Tenant-Id"]
        received_request["internal_token"] = request.headers["X-Internal-Token"]
        return httpx.Response(
            201,
            json={
                "success": True,
                "code": "OK",
                "message": "ok",
                "data": {
                    "ticket_id": "T1001",
                    "ticket_status": "open",
                    "title": make_arguments().title,
                    "category": "complaint",
                    "priority": "high",
                    "related_order_id": "A1001",
                    "created_at": datetime(2026, 7, 12, tzinfo=timezone.utc).isoformat(),
                    "user_visible_summary": "Ticket created.",
                },
                "trace_id": "trace-ticket-client-001",
            },
            headers={TRACE_ID_HEADER: "trace-ticket-client-001"},
            request=request,
        )

    caplog.set_level(logging.INFO, logger="app.services.java_ticket_client")
    settings = Settings(
        java_business_internal_token="token-001",
        java_business_internal_caller="ai-service",
        _env_file=None,
    )
    client = JavaTicketClient(
        base_url="http://java-mock.test",
        timeout_seconds=1,
        settings=settings,
        transport=httpx.MockTransport(handler),
    )
    token = set_trace_id("trace-ticket-client-001")
    context_tokens = set_business_context(user_id="U1001", tenant_id="default")

    try:
        result = client.create_ticket(
            make_arguments(),
            idempotency_key="9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11",
        )
    finally:
        reset_trace_id(token)
        reset_business_context(context_tokens)

    assert received_request["method"] == "POST"
    assert received_request["path"] == "/internal/tickets"
    assert received_request["body"] == {
        "title": make_arguments().title,
        "description": make_arguments().description,
        "category": "complaint",
        "priority": "high",
        "related_order_id": "A1001",
        "source": "ai_agent",
        "confirmation_id": "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11",
    }
    assert received_request["idempotency_key"] == "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11"
    assert received_request["trace_id"] == "trace-ticket-client-001"
    assert received_request["caller"] == "ai-service"
    assert received_request["user_id"] == "U1001"
    assert received_request["tenant_id"] == "default"
    assert received_request["internal_token"] == "token-001"
    assert "upstream_trace_id=trace-ticket-client-001" in caplog.text
    assert result.ticket_id == "T1001"
    assert result.requester_id == "demo_user_001"
    assert result.description == make_arguments().description
    assert result.created_at == datetime(2026, 7, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (httpx.Response(500), "TOOL_UPSTREAM_ERROR"),
        (httpx.Response(400), "TICKET_UPSTREAM_REJECTED"),
        (httpx.Response(201, content=b"not-json"), "TOOL_RESULT_VALIDATION_FAILED"),
    ],
)
def test_java_ticket_client_maps_untrusted_upstream_failures(
    response: httpx.Response,
    code: str,
) -> None:
    client = JavaTicketClient(
        base_url="http://java-mock.test",
        timeout_seconds=1,
        transport=httpx.MockTransport(lambda request: response),
    )

    with pytest.raises(AppException) as exc_info:
        client.create_ticket(
            make_arguments(),
            idempotency_key="confirmation-idempotency-002",
        )

    assert exc_info.value.code == code


def test_java_ticket_client_maps_order_not_support_ticket_to_user_safe_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "success": False,
                "code": "ORDER_NOT_SUPPORT_TICKET",
                "message": "当前订单不支持创建该类工单。",
            },
            request=request,
        )

    client = JavaTicketClient(
        base_url="http://java-mock.test",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AppException) as exc_info:
        client.create_ticket(
            make_arguments(),
            idempotency_key="confirmation-idempotency-003",
        )

    exc = exc_info.value
    assert exc.code == "ORDER_NOT_SUPPORT_TICKET"
    assert exc.message == "当前订单暂不支持创建这类工单，如需帮助可以联系人工客服。"
    assert exc.status_code == 409


def test_java_ticket_client_maps_idempotency_conflict_to_reconfirm_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "success": False,
                "code": "IDEMPOTENCY_KEY_CONFLICT",
                "message": "同一个幂等键不能用于不同的请求参数。",
            },
            request=request,
        )

    client = JavaTicketClient(
        base_url="http://java-mock.test",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AppException) as exc_info:
        client.create_ticket(
            make_arguments(),
            idempotency_key="confirmation-idempotency-004",
        )

    exc = exc_info.value
    assert exc.code == "IDEMPOTENCY_KEY_CONFLICT"
    assert exc.message == "本次提交和已确认的工单请求不一致，请重新确认后再提交。"
    assert exc.status_code == 409
