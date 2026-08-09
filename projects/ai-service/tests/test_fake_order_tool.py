import pytest

from app.core.exceptions import AppException
from app.schemas.tool import OrderStatus, QueryOrderArgs
from app.tools.fake_order_tool import (
    map_java_order_to_query_order_payload,
    map_query_order_error,
    query_order,
    validate_query_order_result,
)
from tests.tool_fakes import FakeOrderLookupClient, make_java_order_payload


def test_query_order_returns_java_business_order_result() -> None:
    client = FakeOrderLookupClient(make_java_order_payload())

    result = query_order(QueryOrderArgs(order_id="A1001"), client=client)

    assert result.order_id == "A1001"
    assert result.order_status == OrderStatus.WAITING_SHIPMENT
    assert result.logistics_message == "商家已接单，等待仓库发货。"
    assert result.can_create_ticket is True
    assert result.source == "java_business_service"
    assert client.calls == ["A1001"]


def test_query_order_does_not_expose_java_customer_id() -> None:
    client = FakeOrderLookupClient(make_java_order_payload(customer_id="C_SECRET"))

    result = query_order(QueryOrderArgs(order_id="A1001"), client=client)
    data = result.model_dump()

    assert "customer_id" not in data


def test_query_order_preserves_client_app_exception() -> None:
    client = FakeOrderLookupClient(
        error=AppException(
            code="ORDER_NOT_FOUND",
            message="订单不存在，请确认订单号是否正确。",
            status_code=404,
        )
    )

    with pytest.raises(AppException) as exc_info:
        query_order(QueryOrderArgs(order_id="A9999"), client=client)

    exc = exc_info.value
    assert exc.code == "ORDER_NOT_FOUND"
    assert exc.message == "订单不存在，请确认订单号是否正确。"
    assert exc.status_code == 404


def test_query_order_maps_unexpected_client_error_to_tool_call_failed() -> None:
    client = FakeOrderLookupClient(error=RuntimeError("unexpected"))

    with pytest.raises(AppException) as exc_info:
        query_order(QueryOrderArgs(order_id="A1001"), client=client)

    exc = exc_info.value
    assert exc.code == "TOOL_CALL_FAILED"
    assert exc.message == "工具调用失败，请稍后重试。"
    assert exc.status_code == 502


def test_map_java_order_to_query_order_payload_drops_unknown_fields() -> None:
    payload = map_java_order_to_query_order_payload(
        make_java_order_payload(customer_id="C9001", internal_note="hidden")
    )

    assert payload == {
        "order_id": "A1001",
        "order_status": "waiting_shipment",
        "payment_status": "paid",
        "logistics_message": "商家已接单，等待仓库发货。",
        "latest_event": "仓库正在准备出库。",
        "can_create_ticket": True,
        "source": "java_business_service",
    }


def test_validate_query_order_result_accepts_mapped_java_result() -> None:
    result = validate_query_order_result(
        {
            "order_id": "A1001",
            "order_status": "waiting_shipment",
            "payment_status": "paid",
            "logistics_message": "商家已接单，等待仓库发货。",
            "latest_event": "仓库正在准备出库。",
            "can_create_ticket": True,
            "source": "java_mock_service",
        }
    )

    assert result.order_id == "A1001"
    assert result.order_status == OrderStatus.WAITING_SHIPMENT
    assert result.source == "java_mock_service"


def test_validate_query_order_result_raises_when_status_is_invalid() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_query_order_result(
            {
                "order_id": "A1001",
                "order_status": "unknown_status",
                "payment_status": "paid",
                "logistics_message": "商家已接单，等待仓库发货。",
                "latest_event": "仓库正在准备出库。",
                "can_create_ticket": True,
                "source": "java_mock_service",
            }
        )

    exc = exc_info.value
    assert exc.code == "TOOL_RESULT_VALIDATION_FAILED"
    assert exc.status_code == 502
    assert exc.details is not None
    assert exc.details[0]["loc"] == ("order_status",)
    assert exc.details[0]["type"] == "enum"
    assert "input" not in exc.details[0]


def test_validate_query_order_result_raises_when_required_field_is_missing() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_query_order_result(
            {
                "order_id": "A1001",
                "order_status": "waiting_shipment",
                "payment_status": "paid",
                "latest_event": "仓库正在准备出库。",
                "can_create_ticket": True,
                "source": "java_mock_service",
            }
        )

    exc = exc_info.value
    assert exc.code == "TOOL_RESULT_VALIDATION_FAILED"
    assert exc.details is not None
    assert exc.details[0]["loc"] == ("logistics_message",)
    assert exc.details[0]["type"] == "missing"


def test_map_query_order_error_keeps_app_exception() -> None:
    app_exception = AppException(
        code="ORDER_NOT_FOUND",
        message="订单不存在，请确认订单号是否正确。",
        status_code=404,
    )

    assert map_query_order_error(app_exception) is app_exception


def test_map_query_order_error_returns_fallback_app_exception() -> None:
    exc = map_query_order_error(RuntimeError("unexpected"))

    assert exc.code == "TOOL_CALL_FAILED"
    assert exc.message == "工具调用失败，请稍后重试。"
    assert exc.status_code == 502
