import json
import logging

import pytest

from app.agents.ticket_agent import (
    LLMTicketIntentClassifier,
    TICKET_INTENT_CLASSIFICATION_PROMPT,
    build_ticket_agent_graph,
    build_ticket_agent_input,
    build_ticket_intent_classification_messages,
    classify_intent_node,
    parse_ticket_intent_classification_json,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from tests.fakes import (
    FakeChatCompletions,
    FakeOpenAICompatibleClient,
    make_usage,
)


class StaticIntentClassifier:
    def __init__(self, intent: str, reason: str = "fake classifier reason") -> None:
        self.intent = intent
        self.reason = reason
        self.messages: list[str] = []

    def classify_intent(self, message: str) -> dict[str, str]:
        self.messages.append(message)
        return {
            "intent": self.intent,
            "reason": self.reason,
        }


def test_build_ticket_intent_classification_messages_include_schema_and_user_message() -> None:
    messages = build_ticket_intent_classification_messages("退款规则是什么？")

    assert messages[0]["role"] == "system"
    assert "只返回合法 JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "JSON Schema:" in messages[1]["content"]
    assert "policy_question" in messages[1]["content"]
    assert "ticket_request" in messages[1]["content"]
    assert "用户消息:\n退款规则是什么？" in messages[1]["content"]


def test_parse_ticket_intent_classification_json_returns_validated_classification() -> None:
    classification = parse_ticket_intent_classification_json(
        json.dumps(
            {
                "intent": "order_query",
                "reason": "  用户在询问订单状态。  ",
            },
            ensure_ascii=False,
        )
    )

    assert classification == {
        "intent": "order_query",
        "reason": "用户在询问订单状态。",
    }


def test_parse_ticket_intent_classification_json_rejects_invalid_intent() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_intent_classification_json(
            json.dumps(
                {
                    "intent": "refund",
                    "reason": "refund 不是 Agent 路由允许的 intent。",
                },
                ensure_ascii=False,
            )
        )

    assert exc_info.value.code == "TICKET_INTENT_LLM_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502


def test_parse_ticket_intent_classification_json_rejects_empty_response() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_intent_classification_json("   ")

    assert exc_info.value.code == "TICKET_INTENT_LLM_EMPTY_RESPONSE"


def test_llm_ticket_intent_classifier_calls_openai_compatible_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.agents.ticket_agent")
    completions = FakeChatCompletions(
        content=json.dumps(
            {
                "intent": "ticket_request",
                "reason": "用户明确要求人工处理投诉。",
            },
            ensure_ascii=False,
        ),
        usage=make_usage(
            prompt_tokens=40,
            completion_tokens=12,
            total_tokens=52,
        ),
    )
    classifier = LLMTicketIntentClassifier(
        Settings(
            llm_api_key="test-key",
            llm_provider="test-provider",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeOpenAICompatibleClient(completions),
    )

    classification = classifier.classify_intent("我要投诉订单 1001")

    assert classification == {
        "intent": "ticket_request",
        "reason": "用户明确要求人工处理投诉。",
    }
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "qwen-test"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert "我要投诉订单 1001" in call["messages"][1]["content"]

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ticket_intent_llm_classification_succeeded provider=test-provider model=qwen-test"
        in message
        and f"prompt_name={TICKET_INTENT_CLASSIFICATION_PROMPT.name}" in message
        and f"prompt_version={TICKET_INTENT_CLASSIFICATION_PROMPT.version}" in message
        and "intent=ticket_request" in message
        and "prompt_tokens=40" in message
        for message in messages
    )
    assert all("test-key" not in message for message in messages)


def test_llm_ticket_intent_classifier_requires_api_key() -> None:
    classifier = LLMTicketIntentClassifier(
        Settings(llm_api_key=None, openai_api_key=None, _env_file=None),
        client=FakeOpenAICompatibleClient(FakeChatCompletions()),
    )

    with pytest.raises(AppException) as exc_info:
        classifier.classify_intent("退款规则是什么？")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"


def test_classify_intent_node_can_use_injected_classifier() -> None:
    classifier = StaticIntentClassifier(
        "smalltalk",
        reason="fake classifier overrides the rule result",
    )

    update = classify_intent_node(
        {"normalized_message": "我的订单 1001 到哪了？"},
        classifier=classifier,
    )

    assert update == {
        "intent": "smalltalk",
        "intent_reason": "fake classifier overrides the rule result",
        "node_history": ["classify_intent"],
    }
    assert classifier.messages == ["我的订单 1001 到哪了？"]


def test_build_ticket_agent_graph_can_route_with_injected_intent_classifier() -> None:
    classifier = StaticIntentClassifier(
        "smalltalk",
        reason="fake classifier selected smalltalk",
    )
    graph = build_ticket_agent_graph(intent_classifier=classifier)

    result = graph.invoke(build_ticket_agent_input("我的订单 1001 到哪了？"))

    assert result["intent"] == "smalltalk"
    assert result["intent_reason"] == "fake classifier selected smalltalk"
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_direct_answer",
    ]
    assert classifier.messages == ["我的订单 1001 到哪了？"]
