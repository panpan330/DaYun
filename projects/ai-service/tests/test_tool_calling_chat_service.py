import json
import logging
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.tool import OrderStatus, PaymentStatus, QueryOrderArgs, QueryOrderResult
from app.schemas.tool_decision import ToolCallCandidate
from app.services.tool_calling_chat_service import (
    ToolCallingChatService,
    build_tool_summary_messages,
)
from tests.fakes import (
    FakeChatCompletions,
    FakeOpenAICompatibleClient,
    make_chat_completion,
    make_tool_call,
)


class SequentialFakeChatCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected model call")

        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_query_order_result() -> QueryOrderResult:
    return QueryOrderResult(
        order_id="A1001",
        order_status=OrderStatus.WAITING_SHIPMENT,
        payment_status=PaymentStatus.PAID,
        logistics_message="商家已接单，等待仓库发货。",
        latest_event="仓库正在准备出库。",
        can_create_ticket=True,
        source="java_mock_service",
    )


def make_service(
    completions: object,
    *,
    query_order_executor: Any = None,
) -> ToolCallingChatService:
    return ToolCallingChatService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        client=FakeOpenAICompatibleClient(completions),
        query_order_executor=query_order_executor,
    )


def test_build_tool_summary_messages_preserves_tool_call_relationship() -> None:
    tool_call = ToolCallCandidate(
        name="query_order",
        arguments={"order_id": "A1001"},
        call_id="call_query_order_001",
    )

    messages = build_tool_summary_messages(
        [{"role": "system", "content": "system"}],
        tool_call,
        make_query_order_result(),
    )

    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_query_order_001",
                "type": "function",
                "function": {
                    "name": "query_order",
                    "arguments": '{"order_id":"A1001"}',
                },
            }
        ],
    }
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_query_order_001"
    assert json.loads(messages[2]["content"]) == {
        "order_id": "A1001",
        "order_status": "waiting_shipment",
        "payment_status": "paid",
        "logistics_message": "商家已接单，等待仓库发货。",
        "latest_event": "仓库正在准备出库。",
        "can_create_ticket": True,
        "source": "java_mock_service",
    }


def test_tool_calling_chat_executes_tool_and_asks_model_to_summarize() -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[
                    make_tool_call(
                        "query_order",
                        {"order_id": "A1001"},
                        call_id="call_query_order_001",
                    )
                ],
            ),
            make_chat_completion(
                "订单 A1001 已付款，商家已接单，仓库正在准备出库。"
            ),
        ]
    )
    executor_calls: list[str] = []

    def query_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
        executor_calls.append(arguments.order_id)
        return make_query_order_result()

    service = make_service(completions, query_order_executor=query_order_executor)

    reply = service.generate_reply("帮我查订单 A1001")

    assert reply == "订单 A1001 已付款，商家已接单，仓库正在准备出库。"
    assert executor_calls == ["A1001"]
    assert len(completions.calls) == 2

    first_call = completions.calls[0]
    assert first_call["model"] == "qwen-test"
    assert first_call["tool_choice"] == "auto"
    assert first_call["tools"][0]["function"]["name"] == "query_order"

    second_call = completions.calls[1]
    assert second_call["tool_choice"] == "auto"
    assistant_tool_call = second_call["messages"][-2]
    tool_result_message = second_call["messages"][-1]
    assert assistant_tool_call["role"] == "assistant"
    assert assistant_tool_call["tool_calls"][0]["id"] == "call_query_order_001"
    assert tool_result_message["role"] == "tool"
    assert tool_result_message["tool_call_id"] == "call_query_order_001"
    assert "customer_id" not in tool_result_message["content"]


def test_tool_calling_chat_logs_tool_execution_stages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[
                    make_tool_call(
                        "query_order",
                        {"order_id": "A1001"},
                        call_id="call_query_order_001",
                    )
                ],
            ),
            make_chat_completion("订单 A1001 正在准备出库。"),
        ]
    )
    service = make_service(
        completions,
        query_order_executor=lambda arguments: make_query_order_result(),
    )
    caplog.set_level(logging.INFO, logger="app.services.tool_calling_chat_service")

    service.generate_reply("帮我查订单 A1001")

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.services.tool_calling_chat_service"
    ]

    assert any(
        message.startswith("tool_execution_started tool_name=query_order")
        for message in messages
    )
    assert any(
        message.startswith("tool_execution_succeeded tool_name=query_order")
        for message in messages
    )


def test_tool_calling_chat_returns_direct_model_reply_without_executing_tool() -> None:
    completions = FakeChatCompletions(content="FastAPI 是一个 Python Web 框架。")
    executor_calls: list[str] = []

    def query_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
        executor_calls.append(arguments.order_id)
        return make_query_order_result()

    service = make_service(completions, query_order_executor=query_order_executor)

    reply = service.generate_reply("FastAPI 是什么？")

    assert reply == "FastAPI 是一个 Python Web 框架。"
    assert executor_calls == []
    assert len(completions.calls) == 1


def test_tool_calling_chat_requires_tool_call_id_before_executing_tool() -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[
                    make_tool_call(
                        "query_order",
                        {"order_id": "A1001"},
                        call_id=None,
                    )
                ],
            )
        ]
    )
    executor_calls: list[str] = []

    def query_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
        executor_calls.append(arguments.order_id)
        return make_query_order_result()

    service = make_service(completions, query_order_executor=query_order_executor)

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("帮我查订单 A1001")

    assert exc_info.value.code == "TOOL_CALL_ID_MISSING"
    assert exc_info.value.status_code == 502
    assert executor_calls == []
    assert len(completions.calls) == 1


def test_tool_calling_chat_preserves_tool_execution_error_and_skips_summary() -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[make_tool_call("query_order", {"order_id": "A9999"})],
            )
        ]
    )

    def query_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
        raise AppException(
            code="ORDER_NOT_FOUND",
            message="订单不存在，请确认订单号是否正确。",
            status_code=404,
        )

    service = make_service(completions, query_order_executor=query_order_executor)

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("帮我查订单 A9999")

    assert exc_info.value.code == "ORDER_NOT_FOUND"
    assert exc_info.value.status_code == 404
    assert len(completions.calls) == 1


def test_tool_calling_chat_rejects_second_model_tool_call_in_current_lesson() -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[make_tool_call("query_order", {"order_id": "A1001"})],
            ),
            make_chat_completion(
                None,
                tool_calls=[make_tool_call("query_order", {"order_id": "A1001"})],
            ),
        ]
    )
    service = make_service(
        completions,
        query_order_executor=lambda arguments: make_query_order_result(),
    )

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("帮我查订单 A1001")

    assert exc_info.value.code == "TOOL_SUMMARY_UNEXPECTED_TOOL_CALL"
    assert exc_info.value.status_code == 502


def test_tool_calling_chat_requires_api_key_before_model_call() -> None:
    completions = FakeChatCompletions(content="不会被调用")
    service = ToolCallingChatService(
        Settings(llm_api_key="", openai_api_key="", _env_file=None),
        client=FakeOpenAICompatibleClient(completions),
    )

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("帮我查订单 A1001")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"
    assert completions.calls == []


def test_tool_calling_chat_maps_second_model_error() -> None:
    completions = SequentialFakeChatCompletions(
        [
            make_chat_completion(
                None,
                tool_calls=[make_tool_call("query_order", {"order_id": "A1001"})],
            ),
            RuntimeError("provider failed"),
        ]
    )
    service = make_service(
        completions,
        query_order_executor=lambda arguments: make_query_order_result(),
    )

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("帮我查订单 A1001")

    assert exc_info.value.code == "LLM_CALL_FAILED"
    assert exc_info.value.status_code == 502


# ---- 工具动态按需注入接入 ----


def test_build_model_tools_respects_selection() -> None:
    service = ToolCallingChatService.__new__(ToolCallingChatService)
    service.settings = Settings(_env_file=None, tool_selection_enabled=True)
    tools = service._build_model_tools(
        [{"role": "user", "content": "订单 A1001 我要退款"}]
    )
    names = [tool["function"]["name"] for tool in tools]
    assert "query_order" in names
    # 写工具即使在 select 结果中，registry 也只暴露 READ 工具（双重保险）
    assert "refund_order" not in names


def test_build_model_tools_disabled_returns_all() -> None:
    service = ToolCallingChatService.__new__(ToolCallingChatService)
    service.settings = Settings(_env_file=None, tool_selection_enabled=False)
    tools = service._build_model_tools([{"role": "user", "content": "你好"}])
    names = [tool["function"]["name"] for tool in tools]
    # 全量 = READ 级别工具（写工具本就不可被模型直接调用）
    assert names == ["query_order"]
