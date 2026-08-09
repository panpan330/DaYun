import pytest
from pydantic import ValidationError

from app.schemas.tool import (
    OrderStatus,
    PaymentStatus,
    QueryOrderArgs,
    QueryOrderResponse,
    QueryOrderResult,
    ToolAccessLevel,
    ToolDefinition,
    get_query_order_args_json_schema,
    get_query_order_result_json_schema,
)


def test_query_order_args_accepts_and_strips_order_id() -> None:
    args = QueryOrderArgs(order_id="  A1001  ")

    assert args.order_id == "A1001"


def test_tool_definition_accepts_backend_owned_tool_metadata() -> None:
    definition = ToolDefinition(
        name="query_order",
        description="查询订单状态。",
        access_level="read",
        requires_confirmation=False,
        enabled=True,
        argument_schema=get_query_order_args_json_schema(),
    )

    assert definition.name == "query_order"
    assert definition.access_level == ToolAccessLevel.READ
    assert definition.requires_confirmation is False
    assert definition.enabled is True


def test_tool_definition_rejects_invalid_tool_name() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolDefinition(
            name="RefundOrder",
            description="非法工具名。",
            access_level="sensitive",
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("name",)
    assert error["type"] == "string_pattern_mismatch"


def test_query_order_args_rejects_empty_order_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        QueryOrderArgs(order_id="   ")

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("order_id",)
    assert error["type"] == "string_too_short"


def test_query_order_args_rejects_order_id_with_spaces_inside() -> None:
    with pytest.raises(ValidationError) as exc_info:
        QueryOrderArgs(order_id="A 1001")

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("order_id",)
    assert error["type"] == "string_pattern_mismatch"


def test_query_order_result_accepts_supported_status_values() -> None:
    result = QueryOrderResult(
        order_id="A1001",
        order_status="waiting_shipment",
        payment_status="paid",
        logistics_message="商家已接单，等待仓库发货。",
        latest_event="仓库正在准备出库。",
        can_create_ticket=True,
    )

    assert result.order_status == OrderStatus.WAITING_SHIPMENT
    assert result.payment_status == PaymentStatus.PAID
    assert result.source == "fake_order_tool"


def test_query_order_result_rejects_unknown_order_status() -> None:
    with pytest.raises(ValidationError) as exc_info:
        QueryOrderResult(
            order_id="A1001",
            order_status="unknown",
            payment_status="paid",
            logistics_message="商家已接单，等待仓库发货。",
            latest_event="仓库正在准备出库。",
            can_create_ticket=True,
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("order_status",)
    assert error["type"] == "enum"


def test_query_order_result_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        QueryOrderResult(
            order_id="A1001",
            order_status="waiting_shipment",
            payment_status="paid",
            logistics_message="商家已接单，等待仓库发货。",
            latest_event="仓库正在准备出库。",
            can_create_ticket=True,
            internal_note="should not leak",
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("internal_note",)
    assert error["type"] == "extra_forbidden"


def test_query_order_response_wraps_result() -> None:
    response = QueryOrderResponse(
        result={
            "order_id": "A1002",
            "order_status": "shipped",
            "payment_status": "paid",
            "logistics_message": "包裹已发出。",
            "latest_event": "快递已揽收。",
            "can_create_ticket": False,
        }
    )

    assert response.result.order_id == "A1002"
    assert response.result.order_status == OrderStatus.SHIPPED


def test_query_order_args_json_schema_contains_expected_fields() -> None:
    schema = get_query_order_args_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"order_id"}
    assert schema["required"] == ["order_id"]
    assert schema["properties"]["order_id"]["pattern"] == r"^[A-Za-z0-9_-]+$"


def test_query_order_result_json_schema_contains_expected_fields() -> None:
    schema = get_query_order_result_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "order_id",
        "order_status",
        "payment_status",
        "logistics_message",
        "latest_event",
        "can_create_ticket",
        "source",
    }
    assert set(schema["required"]) == {
        "order_id",
        "order_status",
        "payment_status",
        "logistics_message",
        "latest_event",
        "can_create_ticket",
    }
