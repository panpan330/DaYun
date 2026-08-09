from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage, ChatMessageRole
from app.schemas.tool_decision import ToolDecisionType
from app.services.tool_decision_service import (
    ToolDecisionService,
    build_tool_decision_messages,
    extract_tool_decision,
    parse_tool_call_arguments,
)
from tests.fakes import (
    FakeChatCompletions as FakeCompletions,
    FakeOpenAICompatibleClient as FakeClient,
    make_chat_completion,
    make_tool_call,
    make_usage,
)


def test_build_tool_decision_messages_includes_decision_rules() -> None:
    messages = build_tool_decision_messages("帮我查订单 A1001")

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "工具调用决策器" in messages[0]["content"]
    assert "query_order" in messages[0]["content"]
    assert "判断规则" in messages[1]["content"]
    assert "帮我查订单 A1001" in messages[1]["content"]


def test_build_tool_decision_messages_keeps_history() -> None:
    history = [
        ChatMessage(role=ChatMessageRole.USER, content="订单 A1001 有问题"),
        ChatMessage(role=ChatMessageRole.ASSISTANT, content="你想查询订单状态吗？"),
    ]

    messages = build_tool_decision_messages("对，查一下", history=history)

    assert messages[1] == {"role": "user", "content": "订单 A1001 有问题"}
    assert messages[2] == {"role": "assistant", "content": "你想查询订单状态吗？"}
    assert messages[3]["role"] == "user"
    assert "对，查一下" in messages[3]["content"]


def test_parse_tool_call_arguments_accepts_json_object_string() -> None:
    assert parse_tool_call_arguments('{"order_id":"A1001"}') == {
        "order_id": "A1001"
    }


def test_parse_tool_call_arguments_rejects_invalid_json() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_tool_call_arguments("{not-json")

    assert exc_info.value.code == "TOOL_ARGUMENTS_INVALID_JSON"
    assert exc_info.value.status_code == 502


def test_parse_tool_call_arguments_rejects_json_array() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_tool_call_arguments('["A1001"]')

    assert exc_info.value.code == "TOOL_ARGUMENTS_INVALID_JSON"
    assert exc_info.value.status_code == 502


def test_extract_tool_decision_returns_direct_reply_when_no_tool_call() -> None:
    completion = make_chat_completion("  请提供订单号后我再帮你查询。  ")

    decision = extract_tool_decision(completion)

    assert decision.decision == ToolDecisionType.ANSWER_DIRECTLY
    assert decision.reply == "请提供订单号后我再帮你查询。"
    assert decision.tool_call is None


def test_extract_tool_decision_returns_validated_tool_call() -> None:
    completion = make_chat_completion(
        None,
        tool_calls=[
            make_tool_call(
                "query_order",
                {"order_id": "  A1001  "},
                call_id="call_query_order_001",
            )
        ],
    )

    decision = extract_tool_decision(completion)

    assert decision.decision == ToolDecisionType.CALL_TOOL
    assert decision.reply is None
    assert decision.tool_call is not None
    assert decision.tool_call.name == "query_order"
    assert decision.tool_call.arguments == {"order_id": "A1001"}
    assert decision.tool_call.call_id == "call_query_order_001"


def test_extract_tool_decision_rejects_unknown_tool() -> None:
    completion = make_chat_completion(
        None,
        tool_calls=[make_tool_call("delete_database", {"order_id": "A1001"})],
    )

    with pytest.raises(AppException) as exc_info:
        extract_tool_decision(completion)

    assert exc_info.value.code == "TOOL_NOT_ALLOWED"
    assert exc_info.value.status_code == 403


def test_extract_tool_decision_rejects_write_tool_without_confirmation() -> None:
    completion = make_chat_completion(
        None,
        tool_calls=[make_tool_call("create_ticket", {"order_id": "A1001"})],
    )

    with pytest.raises(AppException) as exc_info:
        extract_tool_decision(completion)

    assert exc_info.value.code == "TOOL_CONFIRMATION_REQUIRED"
    assert exc_info.value.status_code == 409


def test_extract_tool_decision_rejects_invalid_arguments_schema() -> None:
    completion = make_chat_completion(
        None,
        tool_calls=[make_tool_call("query_order", {"order_id": "A 1001"})],
    )

    with pytest.raises(AppException) as exc_info:
        extract_tool_decision(completion)

    assert exc_info.value.code == "TOOL_ARGUMENTS_VALIDATION_FAILED"
    assert exc_info.value.status_code == 502
    assert exc_info.value.details[0]["loc"] == ("order_id",)


def test_extract_tool_decision_rejects_multiple_tool_calls() -> None:
    completion = make_chat_completion(
        None,
        tool_calls=[
            make_tool_call("query_order", {"order_id": "A1001"}, call_id="call_1"),
            make_tool_call("query_order", {"order_id": "A1002"}, call_id="call_2"),
        ],
    )

    with pytest.raises(AppException) as exc_info:
        extract_tool_decision(completion)

    assert exc_info.value.code == "TOOL_DECISION_TOO_MANY_CALLS"
    assert exc_info.value.status_code == 502


def test_extract_tool_decision_rejects_missing_message_shape() -> None:
    with pytest.raises(AppException) as exc_info:
        extract_tool_decision(SimpleNamespace(choices=[]))

    assert exc_info.value.code == "LLM_BAD_RESPONSE"
    assert exc_info.value.status_code == 502


def test_tool_decision_service_calls_model_with_tools() -> None:
    completions = FakeCompletions(
        content=None,
        tool_calls=[make_tool_call("query_order", {"order_id": "A1001"})],
        usage=make_usage(prompt_tokens=30, completion_tokens=5, total_tokens=35),
    )
    service = ToolDecisionService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        client=FakeClient(completions),
    )

    decision = service.decide("帮我查订单 A1001")

    assert decision.decision == ToolDecisionType.CALL_TOOL
    assert decision.tool_call is not None
    assert decision.tool_call.arguments == {"order_id": "A1001"}
    call = completions.last_call
    assert call["model"] == "qwen-test"
    assert call["tool_choice"] == "auto"
    assert call["tools"][0]["type"] == "function"
    assert call["tools"][0]["function"]["name"] == "query_order"
    assert call["tools"][0]["function"]["strict"] is True
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"


def test_tool_decision_service_returns_direct_reply() -> None:
    completions = FakeCompletions(content="请提供订单号后我再帮你查询。")
    service = ToolDecisionService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        client=FakeClient(completions),
    )

    decision = service.decide("帮我查一下订单")

    assert decision.decision == ToolDecisionType.ANSWER_DIRECTLY
    assert decision.reply == "请提供订单号后我再帮你查询。"
    assert decision.tool_call is None


def test_tool_decision_service_requires_api_key() -> None:
    completions = FakeCompletions(content="不会被调用")
    service = ToolDecisionService(
        Settings(llm_api_key="", openai_api_key="", _env_file=None),
        client=FakeClient(completions),
    )

    with pytest.raises(AppException) as exc_info:
        service.decide("帮我查订单 A1001")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"
    assert exc_info.value.status_code == 500
    assert completions.calls == []


def test_tool_decision_service_maps_provider_errors() -> None:
    service = ToolDecisionService(
        Settings(llm_api_key="test-key", _env_file=None),
        client=FakeClient(FakeCompletions(error=RuntimeError("provider failed"))),
    )

    with pytest.raises(AppException) as exc_info:
        service.decide("帮我查订单 A1001")

    assert exc_info.value.code == "LLM_CALL_FAILED"
    assert exc_info.value.status_code == 502
