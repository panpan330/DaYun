import logging

import pytest

from app.agents.ticket_agent import (
    ModelOutputFallbackTicketFieldExtractor,
    ModelOutputFallbackTicketIntentClassifier,
    build_ticket_agent_graph_for_model_mode,
    build_ticket_agent_input,
    classify_ticket_agent_model_output_failure,
    extract_ticket_fields_node,
    parse_ticket_intent_classification_json,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient


class FailingIntentClassifier:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def classify_intent(self, message: str) -> dict[str, str]:
        raise self.exc


class FailingFieldExtractor:
    extraction_source = "llm"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def extract_fields(self, state: dict[str, object]) -> dict[str, object]:
        raise self.exc


def test_classify_model_output_failure_marks_empty_response_as_fallback() -> None:
    failure = classify_ticket_agent_model_output_failure(
        AppException(
            code="TICKET_INTENT_LLM_EMPTY_RESPONSE",
            message="模型没有返回内容",
            status_code=502,
        )
    )

    assert failure.code == "TICKET_INTENT_LLM_EMPTY_RESPONSE"
    assert failure.kind == "empty_response"
    assert failure.action == "fallback_to_rule_based"
    assert failure.retryable is True


def test_classify_model_output_failure_distinguishes_invalid_json() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_intent_classification_json("{invalid-json")

    failure = classify_ticket_agent_model_output_failure(exc_info.value)

    assert failure.code == "TICKET_INTENT_LLM_VALIDATION_FAILED"
    assert failure.kind == "invalid_json"
    assert failure.action == "fallback_to_rule_based"
    assert failure.retryable is True


def test_classify_model_output_failure_keeps_configuration_error_visible() -> None:
    failure = classify_ticket_agent_model_output_failure(
        AppException(
            code="LLM_API_KEY_MISSING",
            message="LLM API key 未配置",
            status_code=500,
        )
    )

    assert failure.kind == "configuration_error"
    assert failure.action == "raise_error"
    assert failure.retryable is False


def test_intent_classifier_can_fallback_to_rule_based_when_llm_output_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="app.agents.ticket_agent")
    classifier = ModelOutputFallbackTicketIntentClassifier(
        FailingIntentClassifier(
            AppException(
                code="TICKET_INTENT_LLM_EMPTY_RESPONSE",
                message="模型没有返回内容",
                status_code=502,
            )
        )
    )

    result = classifier.classify_intent("我要投诉，帮我创建工单")

    assert result["intent"] == "ticket_request"
    assert "ticket_agent_model_output_fallback component=intent_classifier" in caplog.text
    assert "kind=empty_response" in caplog.text


def test_intent_classifier_does_not_hide_configuration_errors() -> None:
    classifier = ModelOutputFallbackTicketIntentClassifier(
        FailingIntentClassifier(
            AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置",
                status_code=500,
            )
        )
    )

    with pytest.raises(AppException) as exc_info:
        classifier.classify_intent("我要投诉，帮我创建工单")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"


def test_field_extractor_can_fallback_to_rule_based_and_mark_source() -> None:
    extractor = ModelOutputFallbackTicketFieldExtractor(
        FailingFieldExtractor(
            AppException(
                code="TICKET_FIELD_LLM_VALIDATION_FAILED",
                message="模型字段结果不符合 schema",
                status_code=502,
                details=[{"type": "extra_forbidden"}],
            )
        )
    )

    update = extract_ticket_fields_node(
        {
            "normalized_message": "订单 A2001 商品破损，帮我投诉处理",
            "ticket_need_source": "explicit_user_request",
        },
        extractor=extractor,
    )

    assert update["ticket_fields"]["issue_type"] == "complaint"
    assert update["ticket_fields"]["order_id"] == "A2001"
    assert update["ticket_field_extraction_source"] == "llm_fallback_rule_based"


def test_real_llm_graph_can_enable_model_output_fallback() -> None:
    completions = FakeChatCompletions(content="{invalid-json")
    graph = build_ticket_agent_graph_for_model_mode(
        mode="real_llm",
        settings=Settings(
            llm_api_key="test-key",
            llm_provider="test-provider",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeOpenAICompatibleClient(completions),
        enable_model_output_fallback=True,
    )

    result = graph.invoke(build_ticket_agent_input("订单 A2001 商品破损，帮我投诉处理"))

    assert result["intent"] == "ticket_request"
    assert result["ticket_fields"]["issue_type"] == "complaint"
    assert result["ticket_field_extraction_source"] == "llm_fallback_rule_based"
    assert len(completions.calls) == 2
