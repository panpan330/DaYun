import json
import logging

import pytest

from app.agents.ticket_agent import (
    LLMTicketFieldExtractor,
    TICKET_FIELD_EXTRACTION_PROMPT,
    build_ticket_agent_graph,
    build_ticket_agent_input,
    build_ticket_field_extraction_messages,
    extract_ticket_fields_node,
    parse_ticket_field_extraction_json,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from tests.fakes import (
    FakeChatCompletions,
    FakeOpenAICompatibleClient,
    make_usage,
)


def make_complete_llm_fields() -> dict[str, object]:
    return {
        "issue_type": "complaint",
        "order_id": "A2001",
        "description": "商品破损，用户希望客服处理订单 A2001。",
        "user_request": "人工处理商品破损投诉",
        "urgency": "high",
        "need_human_review": True,
    }


class StaticFieldExtractor:
    extraction_source = "llm"

    def __init__(self, fields: dict[str, object]) -> None:
        self.fields = fields
        self.states: list[dict[str, object]] = []

    def extract_fields(self, state: dict[str, object]) -> dict[str, object]:
        self.states.append(dict(state))
        return self.fields


def test_build_ticket_field_extraction_messages_include_schema_context_and_message() -> None:
    messages = build_ticket_field_extraction_messages(
        {
            "normalized_message": "会员等级政策是什么？",
            "intent": "policy_question",
            "ticket_need_source": "rag_no_context",
            "rag_answer_status": "no_context",
            "rag_no_context_reason": "知识库没有找到会员等级政策。",
        }
    )

    assert messages[0]["role"] == "system"
    assert "工单字段提取器" in messages[0]["content"]
    assert "不要输出 should_create_ticket" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "JSON Schema:" in messages[1]["content"]
    assert "issue_type" in messages[1]["content"]
    assert "policy_gap" in messages[1]["content"]
    assert "ticket_need_source" in messages[1]["content"]
    assert "rag_no_context" in messages[1]["content"]
    assert "用户消息:\n会员等级政策是什么？" in messages[1]["content"]


def test_parse_ticket_field_extraction_json_returns_validated_fields() -> None:
    fields = parse_ticket_field_extraction_json(
        json.dumps(
            {
                "issue_type": "complaint",
                "order_id": "  A2001  ",
                "description": "  商品破损，包装也有明显损坏。  ",
                "user_request": "  人工处理投诉  ",
                "urgency": "high",
                "need_human_review": True,
            },
            ensure_ascii=False,
        )
    )

    assert fields == {
        "issue_type": "complaint",
        "order_id": "A2001",
        "description": "商品破损，包装也有明显损坏。",
        "user_request": "人工处理投诉",
        "urgency": "high",
        "need_human_review": True,
    }


def test_parse_ticket_field_extraction_json_rejects_invalid_issue_type() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json(
            json.dumps(
                {
                    "issue_type": "after_sale",
                    "order_id": "A2001",
                    "description": "商品破损。",
                    "user_request": "人工处理",
                    "urgency": "high",
                    "need_human_review": True,
                },
                ensure_ascii=False,
            )
        )

    assert exc_info.value.code == "TICKET_FIELD_LLM_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502


def test_parse_ticket_field_extraction_json_rejects_empty_response() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json("   ")

    assert exc_info.value.code == "TICKET_FIELD_LLM_EMPTY_RESPONSE"


def test_llm_ticket_field_extractor_calls_openai_compatible_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.agents.ticket_agent")
    completions = FakeChatCompletions(
        content=json.dumps(make_complete_llm_fields(), ensure_ascii=False),
        usage=make_usage(
            prompt_tokens=80,
            completion_tokens=30,
            total_tokens=110,
        ),
    )
    extractor = LLMTicketFieldExtractor(
        Settings(
            llm_api_key="test-key",
            llm_provider="test-provider",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeOpenAICompatibleClient(completions),
    )

    fields = extractor.extract_fields(
        {
            "normalized_message": "商品破损，订单 A2001，帮我投诉处理",
            "intent": "ticket_request",
            "ticket_need_source": "explicit_user_request",
        }
    )

    assert fields == make_complete_llm_fields()
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "qwen-test"
    assert call["response_format"] == {"type": "json_object"}
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert "商品破损，订单 A2001，帮我投诉处理" in call["messages"][1]["content"]

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "ticket_field_llm_extraction_succeeded provider=test-provider model=qwen-test"
        in message
        and f"prompt_name={TICKET_FIELD_EXTRACTION_PROMPT.name}" in message
        and f"prompt_version={TICKET_FIELD_EXTRACTION_PROMPT.version}" in message
        and "issue_type=complaint" in message
        and "has_order_id=True" in message
        and "prompt_tokens=80" in message
        for message in messages
    )
    assert all("test-key" not in message for message in messages)


def test_llm_ticket_field_extractor_requires_api_key() -> None:
    extractor = LLMTicketFieldExtractor(
        Settings(llm_api_key=None, openai_api_key=None, _env_file=None),
        client=FakeOpenAICompatibleClient(FakeChatCompletions()),
    )

    with pytest.raises(AppException) as exc_info:
        extractor.extract_fields({"normalized_message": "商品破损，帮我处理"})

    assert exc_info.value.code == "LLM_API_KEY_MISSING"


def test_extract_ticket_fields_node_can_use_injected_extractor() -> None:
    extractor = StaticFieldExtractor(make_complete_llm_fields())

    update = extract_ticket_fields_node(
        {
            "normalized_message": "商品破损，帮我处理",
            "ticket_need_source": "explicit_user_request",
        },
        extractor=extractor,
    )

    assert update["ticket_fields"] == make_complete_llm_fields()
    assert update["missing_ticket_fields"] == []
    assert update["ticket_fields_complete"] is True
    assert update["ticket_field_extraction_source"] == "llm"
    assert update["node_history"] == ["extract_ticket_fields"]
    assert extractor.states[0]["normalized_message"] == "商品破损，帮我处理"


def test_build_ticket_agent_graph_can_extract_fields_with_injected_extractor() -> None:
    extractor = StaticFieldExtractor(make_complete_llm_fields())
    graph = build_ticket_agent_graph(field_extractor=extractor)

    result = graph.invoke(build_ticket_agent_input("商品破损，帮我处理"))

    assert result["intent"] == "ticket_request"
    assert result["ticket_fields"] == make_complete_llm_fields()
    assert result["missing_ticket_fields"] == []
    assert result["ticket_fields_complete"] is True
    assert result["ticket_field_extraction_source"] == "llm"
    assert result["pending_ticket_confirmation"]["ticket_fields"] == make_complete_llm_fields()
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "decide_ticket_need",
        "extract_ticket_fields",
        "request_ticket_confirmation",
    ]
