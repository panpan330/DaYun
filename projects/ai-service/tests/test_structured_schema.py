import pytest
from pydantic import ValidationError

from app.schemas.structured import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TicketExtraction,
    TicketIntent,
    TicketUrgency,
    get_ticket_extraction_json_schema,
)


def test_structured_output_request_accepts_message() -> None:
    request = StructuredOutputRequest(message="订单 A1001 还没发货")

    assert request.message == "订单 A1001 还没发货"


def test_structured_output_request_rejects_empty_message() -> None:
    with pytest.raises(ValidationError) as exc_info:
        StructuredOutputRequest(message="")

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("message",)
    assert error["type"] == "string_too_short"


def test_ticket_extraction_accepts_supported_values() -> None:
    extraction = TicketExtraction(
        intent="logistics",
        order_id="  A1001  ",
        summary="用户询问订单物流状态",
        urgency="normal",
        need_human_review=False,
    )

    assert extraction.intent == TicketIntent.LOGISTICS
    assert extraction.order_id == "A1001"
    assert extraction.summary == "用户询问订单物流状态"
    assert extraction.urgency == TicketUrgency.NORMAL
    assert extraction.need_human_review is False


def test_ticket_extraction_turns_blank_order_id_into_none() -> None:
    extraction = TicketExtraction(
        intent="unknown",
        order_id="   ",
        summary="用户没有提供订单号",
        urgency="low",
        need_human_review=True,
    )

    assert extraction.order_id is None


def test_ticket_extraction_rejects_unknown_intent() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TicketExtraction(
            intent="hack",
            order_id=None,
            summary="用户想取消订单",
            urgency="normal",
            need_human_review=False,
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("intent",)
    assert error["type"] == "enum"


def test_ticket_extraction_rejects_missing_summary() -> None:
    with pytest.raises(ValidationError) as exc_info:
        TicketExtraction(
            intent="refund",
            order_id="A1001",
            urgency="high",
            need_human_review=True,
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("summary",)
    assert error["type"] == "missing"


def test_structured_output_response_wraps_extraction() -> None:
    response = StructuredOutputResponse(
        extraction={
            "intent": "refund",
            "order_id": "A1001",
            "summary": "用户申请退款",
            "urgency": "normal",
            "need_human_review": False,
        }
    )

    assert response.extraction.intent == TicketIntent.REFUND
    assert response.extraction.order_id == "A1001"


def test_ticket_extraction_json_schema_contains_expected_fields() -> None:
    schema = get_ticket_extraction_json_schema()

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {
        "intent",
        "order_id",
        "summary",
        "urgency",
        "need_human_review",
    }
    assert set(schema["required"]) == {
        "intent",
        "summary",
    }
