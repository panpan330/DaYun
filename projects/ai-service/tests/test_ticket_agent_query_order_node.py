import logging

import pytest

from app.agents.ticket_agent import (
    TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE,
    TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE,
    build_ticket_agent_graph,
    build_ticket_agent_input,
    classify_ticket_order_query_failure,
    query_order_node,
)
from app.core.exceptions import AppException
from app.schemas.tool import QueryOrderArgs, QueryOrderResult


def make_query_order_result(order_id: str = "A1001") -> QueryOrderResult:
    return QueryOrderResult(
        order_id=order_id,
        order_status="waiting_shipment",
        payment_status="paid",
        logistics_message="商家已接单，等待仓库发货。",
        latest_event="仓库正在准备出库。",
        can_create_ticket=True,
        source="java_mock_service",
    )


class RecordingOrderQueryExecutor:
    def __init__(
        self,
        *,
        result: QueryOrderResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[QueryOrderArgs] = []

    def __call__(self, arguments: QueryOrderArgs) -> QueryOrderResult:
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result or make_query_order_result(arguments.order_id)


def test_query_order_node_calls_executor_and_writes_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.agents.ticket_agent")
    executor = RecordingOrderQueryExecutor()

    update = query_order_node(
        {"normalized_message": "我的订单 A1001 到哪了？"},
        order_query_executor=executor,
    )

    assert [call.order_id for call in executor.calls] == ["A1001"]
    assert update["order_query_order_id"] == "A1001"
    assert update["order_query_status"] == "succeeded"
    assert update["order_query_result"]["order_id"] == "A1001"
    assert update["order_query_result"]["source"] == "java_mock_service"
    assert update["order_query_error_kind"] is None
    assert update["order_query_error_action"] is None
    assert update["order_query_retryable"] is None
    assert update["order_query_error_status_code"] is None
    assert "查询到订单 A1001" in update["final_answer"]
    assert "订单状态：待发货" in update["final_answer"]
    assert update["node_history"] == ["query_order"]
    assert "ticket_agent_query_order_succeeded order_id=A1001" in caplog.text


def test_query_order_node_asks_for_order_id_when_message_has_no_order_id() -> None:
    executor = RecordingOrderQueryExecutor()

    update = query_order_node(
        {"normalized_message": "我的订单到哪了？"},
        order_query_executor=executor,
    )

    assert executor.calls == []
    assert update["order_query_order_id"] is None
    assert update["order_query_status"] == "missing_order_id"
    assert update["order_query_error_code"] == "ORDER_ID_REQUIRED"
    assert update["order_query_error_kind"] == "missing_order_id"
    assert update["order_query_error_action"] == "ask_user_for_order_id"
    assert update["order_query_retryable"] is False
    assert update["order_query_error_status_code"] is None
    assert "请提供要查询的订单号" in update["final_answer"]
    assert update["node_history"] == ["query_order"]


def test_classify_ticket_order_query_failure_maps_known_tool_errors() -> None:
    not_found = classify_ticket_order_query_failure(
        AppException(
            code="ORDER_NOT_FOUND",
            message="订单不存在，请确认订单号是否正确。",
            status_code=404,
        )
    )
    assert not_found.kind == "not_found"
    assert not_found.action == "ask_user_to_check_order_id"
    assert not_found.retryable is False
    assert not_found.status_code == 404

    timeout = classify_ticket_order_query_failure(
        AppException(
            code="TOOL_TIMEOUT",
            message="订单查询工具调用超时，请稍后重试。",
            status_code=504,
        )
    )
    assert timeout.kind == "timeout"
    assert timeout.action == "retry_later"
    assert timeout.retryable is True
    assert timeout.status_code == 504

    upstream = classify_ticket_order_query_failure(
        AppException(
            code="TOOL_UPSTREAM_ERROR",
            message="订单查询服务暂时不可用，请稍后重试。",
            status_code=502,
        )
    )
    assert upstream.kind == "upstream_error"
    assert upstream.action == "retry_later"
    assert upstream.retryable is True


def test_classify_ticket_order_query_failure_hides_result_validation_details() -> None:
    failure = classify_ticket_order_query_failure(
        AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="工具返回结果校验失败：internal field mismatch。",
            status_code=502,
        )
    )

    assert failure.kind == "result_validation"
    assert failure.action == "investigate_system"
    assert failure.retryable is False
    assert failure.message == TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE
    assert "internal field mismatch" not in failure.message


def test_classify_ticket_order_query_failure_maps_unknown_exception_to_safe_failure() -> None:
    failure = classify_ticket_order_query_failure(
        RuntimeError("database password leaked in stack trace")
    )

    assert failure.code == "TOOL_CALL_FAILED"
    assert failure.kind == "unknown_error"
    assert failure.action == "retry_later"
    assert failure.retryable is True
    assert failure.message == TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE
    assert "database password" not in failure.message


def test_query_order_node_returns_safe_state_when_tool_raises_app_exception() -> None:
    executor = RecordingOrderQueryExecutor(
        error=AppException(
            code="ORDER_NOT_FOUND",
            message="订单不存在，请确认订单号是否正确。",
            status_code=404,
        )
    )

    update = query_order_node(
        {"normalized_message": "帮我查订单 A9999"},
        order_query_executor=executor,
    )

    assert [call.order_id for call in executor.calls] == ["A9999"]
    assert update["order_query_order_id"] == "A9999"
    assert update["order_query_status"] == "failed"
    assert update["order_query_error_code"] == "ORDER_NOT_FOUND"
    assert update["order_query_error_kind"] == "not_found"
    assert update["order_query_error_action"] == "ask_user_to_check_order_id"
    assert update["agent_error_code"] == "ORDER_NOT_FOUND"
    assert update["agent_error_node"] == "query_order"
    assert update["order_query_retryable"] is False
    assert update["order_query_error_status_code"] == 404
    assert update["fallback_used"] is True
    assert update["final_answer"] == "订单不存在，请确认订单号是否正确。"


def test_query_order_node_marks_timeout_as_retryable() -> None:
    executor = RecordingOrderQueryExecutor(
        error=AppException(
            code="TOOL_TIMEOUT",
            message="订单查询工具调用超时，请稍后重试。",
            status_code=504,
        )
    )

    update = query_order_node(
        {"normalized_message": "帮我查订单 A_TIMEOUT"},
        order_query_executor=executor,
    )

    assert update["order_query_order_id"] == "A_TIMEOUT"
    assert update["order_query_status"] == "failed"
    assert update["order_query_error_code"] == "TOOL_TIMEOUT"
    assert update["order_query_error_kind"] == "timeout"
    assert update["order_query_error_action"] == "retry_later"
    assert update["order_query_retryable"] is True
    assert update["order_query_error_status_code"] == 504
    assert update["final_answer"] == "订单查询工具调用超时，请稍后重试。"


def test_query_order_node_hides_tool_result_validation_details() -> None:
    executor = RecordingOrderQueryExecutor(
        error=AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="工具返回结果校验失败：internal field mismatch。",
            status_code=502,
        )
    )

    update = query_order_node(
        {"normalized_message": "帮我查订单 A1001"},
        order_query_executor=executor,
    )

    assert update["order_query_order_id"] == "A1001"
    assert update["order_query_status"] == "failed"
    assert update["order_query_error_code"] == "TOOL_RESULT_VALIDATION_FAILED"
    assert update["order_query_error_kind"] == "result_validation"
    assert update["order_query_error_action"] == "investigate_system"
    assert update["order_query_retryable"] is False
    assert update["final_answer"] == TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE
    assert "internal field mismatch" not in update["final_answer"]


def test_query_order_node_hides_unknown_exception_details() -> None:
    executor = RecordingOrderQueryExecutor(
        error=RuntimeError("database password leaked in stack trace")
    )

    update = query_order_node(
        {"normalized_message": "帮我查订单 A1001"},
        order_query_executor=executor,
    )

    assert update["order_query_order_id"] == "A1001"
    assert update["order_query_status"] == "failed"
    assert update["order_query_error_code"] == "TOOL_CALL_FAILED"
    assert update["order_query_error_kind"] == "unknown_error"
    assert update["order_query_error_action"] == "retry_later"
    assert update["order_query_retryable"] is True
    assert update["order_query_error_status_code"] == 502
    assert update["final_answer"] == TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE
    assert "database password" not in update["final_answer"]


def test_ticket_agent_graph_can_query_order_with_injected_executor() -> None:
    executor = RecordingOrderQueryExecutor()
    graph = build_ticket_agent_graph(order_query_executor=executor)

    result = graph.invoke(build_ticket_agent_input("我的订单 A1001 到哪了？"))

    assert [call.order_id for call in executor.calls] == ["A1001"]
    assert result["intent"] == "order_query"
    assert result["order_query_status"] == "succeeded"
    assert result["order_query_result"]["order_id"] == "A1001"
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "query_order",
    ]
