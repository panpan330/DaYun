from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketExtraction, TicketIntent, TicketUrgency
from app.services.langchain_structured_output_service import (
    LangChainStructuredOutputService,
    build_langchain_ticket_extraction_messages,
    validate_langchain_ticket_extraction,
)


class FakeStructuredRunnable:
    def __init__(
        self,
        result: Any | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else TicketExtraction(
                intent=TicketIntent.LOGISTICS,
                order_id="A1001",
                summary="用户询问订单物流状态",
                urgency=TicketUrgency.NORMAL,
                need_human_review=False,
            )
        )
        self.error = error
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.result


class FakeLangChainModel:
    def __init__(self, runnable: FakeStructuredRunnable) -> None:
        self.runnable = runnable
        self.structured_output_calls: list[dict[str, Any]] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredRunnable:
        self.structured_output_calls.append({"schema": schema, **kwargs})
        return self.runnable


def test_build_langchain_ticket_extraction_messages_include_schema() -> None:
    messages = build_langchain_ticket_extraction_messages("订单 A1001 还没发货")

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "JSON" in str(messages[0].content)
    assert "JSON Schema" in str(messages[1].content)
    assert '"intent"' in str(messages[1].content)
    assert "订单 A1001 还没发货" in str(messages[1].content)


def test_validate_langchain_ticket_extraction_accepts_pydantic_model() -> None:
    extraction = TicketExtraction(
        intent=TicketIntent.COMPLAINT,
        order_id="A1001",
        summary="用户投诉订单迟迟未发货",
        urgency=TicketUrgency.HIGH,
        need_human_review=True,
    )

    assert validate_langchain_ticket_extraction(extraction) is extraction


def test_validate_langchain_ticket_extraction_accepts_dict() -> None:
    extraction = validate_langchain_ticket_extraction(
        {
            "intent": "refund",
            "order_id": "A1001",
            "summary": "用户申请退款",
            "urgency": "high",
            "need_human_review": True,
        }
    )

    assert extraction.intent == TicketIntent.REFUND
    assert extraction.order_id == "A1001"
    assert extraction.need_human_review is True


def test_validate_langchain_ticket_extraction_rejects_invalid_data() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_langchain_ticket_extraction({"intent": "cancel"})

    assert exc_info.value.code == "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502
    assert exc_info.value.details


def test_langchain_structured_output_service_uses_with_structured_output() -> None:
    runnable = FakeStructuredRunnable()
    model = FakeLangChainModel(runnable)
    service = LangChainStructuredOutputService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=model,
    )

    extraction = service.extract_ticket("订单 A1001 还没发货")

    assert extraction.intent == TicketIntent.LOGISTICS
    assert extraction.order_id == "A1001"
    assert model.structured_output_calls == [
        {"schema": TicketExtraction, "method": "json_mode"}
    ]
    assert len(runnable.calls) == 1
    assert isinstance(runnable.calls[0][0], SystemMessage)
    assert isinstance(runnable.calls[0][1], HumanMessage)


def test_langchain_structured_output_service_reuses_structured_model() -> None:
    runnable = FakeStructuredRunnable()
    model = FakeLangChainModel(runnable)
    service = LangChainStructuredOutputService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=model,
    )

    service.extract_ticket("订单 A1001 还没发货")
    service.extract_ticket("订单 A1002 我想退款")

    assert len(model.structured_output_calls) == 1
    assert len(runnable.calls) == 2


def test_langchain_structured_output_service_requires_api_key() -> None:
    service = LangChainStructuredOutputService(
        Settings(_env_file=None),
        model=FakeLangChainModel(FakeStructuredRunnable()),
    )

    with pytest.raises(AppException) as exc_info:
        service.extract_ticket("订单 A1001 还没发货")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"
    assert exc_info.value.status_code == 500


def test_langchain_structured_output_service_maps_invalid_structured_result() -> None:
    service = LangChainStructuredOutputService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=FakeLangChainModel(FakeStructuredRunnable(result={"intent": "cancel"})),
    )

    with pytest.raises(AppException) as exc_info:
        service.extract_ticket("订单 A1001 还没发货")

    assert exc_info.value.code == "STRUCTURED_OUTPUT_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502


def test_langchain_structured_output_service_maps_unknown_model_error() -> None:
    service = LangChainStructuredOutputService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=FakeLangChainModel(
            FakeStructuredRunnable(error=RuntimeError("provider failed"))
        ),
    )

    with pytest.raises(AppException) as exc_info:
        service.extract_ticket("订单 A1001 还没发货")

    assert exc_info.value.code == "LLM_CALL_FAILED"
    assert exc_info.value.status_code == 502
