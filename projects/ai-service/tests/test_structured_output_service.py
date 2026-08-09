import logging
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketIntent, TicketUrgency
from app.services.structured_output_service import (
    StructuredOutputService,
    build_ticket_extraction_messages,
    parse_ticket_extraction_json,
)
from tests.fakes import (
    FakeChatCompletions as FakeCompletions,
    FakeOpenAICompatibleClient as FakeClient,
)


DEFAULT_TICKET_EXTRACTION_JSON = (
    '{"intent":"logistics","order_id":"A1001",'
    '"summary":"用户询问订单物流状态","urgency":"normal",'
    '"need_human_review":false}'
)


def test_build_ticket_extraction_messages_includes_json_schema() -> None:
    messages = build_ticket_extraction_messages("订单 A1001 还没发货")

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "JSON" in messages[0]["content"]
    assert "JSON Schema" in messages[1]["content"]
    assert '"intent"' in messages[1]["content"]
    assert '"summary"' in messages[1]["content"]
    assert "订单 A1001 还没发货" in messages[1]["content"]


def test_parse_ticket_extraction_json_accepts_valid_json() -> None:
    extraction = parse_ticket_extraction_json(
        """
        {
          "intent": "refund",
          "order_id": "A1001",
          "summary": "用户申请退款",
          "urgency": "high",
          "need_human_review": true
        }
        """
    )

    assert extraction.intent == TicketIntent.REFUND
    assert extraction.order_id == "A1001"
    assert extraction.summary == "用户申请退款"
    assert extraction.urgency == TicketUrgency.HIGH
    assert extraction.need_human_review is True


def test_parse_ticket_extraction_json_rejects_invalid_json() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_extraction_json("不是 JSON")

    assert exc_info.value.code == "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502
    assert exc_info.value.details


def test_parse_ticket_extraction_json_rejects_schema_mismatch() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_extraction_json(
            """
            {
              "intent": "hack",
              "summary": "用户想取消订单",
              "urgency": "normal",
              "need_human_review": false
            }
            """
        )

    assert exc_info.value.code == "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502
    assert exc_info.value.details


def test_parse_ticket_extraction_json_rejects_empty_content() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_extraction_json("   ")

    assert exc_info.value.code == "STRUCTURED_OUTPUT_EMPTY"
    assert exc_info.value.status_code == 502


def test_structured_output_service_calls_json_mode() -> None:
    completions = FakeCompletions(content=DEFAULT_TICKET_EXTRACTION_JSON)
    service = StructuredOutputService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        client=FakeClient(completions),
    )

    extraction = service.extract_ticket("订单 A1001 还没发货")

    assert extraction.intent == TicketIntent.LOGISTICS
    assert extraction.order_id == "A1001"
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "qwen-test"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"


def test_structured_output_service_requires_api_key() -> None:
    completions = FakeCompletions(content=DEFAULT_TICKET_EXTRACTION_JSON)
    service = StructuredOutputService(
        Settings(llm_api_key="", openai_api_key="", _env_file=None),
        client=FakeClient(completions),
    )

    with pytest.raises(AppException) as exc_info:
        service.extract_ticket("订单 A1001 还没发货")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"
    assert exc_info.value.status_code == 500
    assert completions.calls == []


def test_structured_output_service_maps_provider_errors() -> None:
    service = StructuredOutputService(
        Settings(llm_api_key="test-key", _env_file=None),
        client=FakeClient(FakeCompletions(error=RuntimeError("provider failed"))),
    )

    with pytest.raises(AppException) as exc_info:
        service.extract_ticket("订单 A1001 还没发货")

    assert exc_info.value.code == "LLM_CALL_FAILED"
    assert exc_info.value.status_code == 502


def test_structured_output_service_logs_success_without_sensitive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.services.structured_output_service")
    completions = FakeCompletions(
        content=DEFAULT_TICKET_EXTRACTION_JSON,
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=12,
            total_tokens=33,
        ),
    )
    service = StructuredOutputService(
        Settings(
            llm_api_key="test-key",
            llm_provider="test-provider",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeClient(completions),
    )

    extraction = service.extract_ticket("订单 A1001 还没发货")

    assert extraction.intent == TicketIntent.LOGISTICS
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "structured_ticket_extraction_succeeded provider=test-provider model=qwen-test"
        in message
        and "intent=logistics" in message
        and "prompt_tokens=21" in message
        and "completion_tokens=12" in message
        and "total_tokens=33" in message
        for message in messages
    )
    assert all("订单 A1001 还没发货" not in message for message in messages)
    assert all("test-key" not in message for message in messages)
