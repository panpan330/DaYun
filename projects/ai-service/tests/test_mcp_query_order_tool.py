import asyncio
import json

import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from app.core.exceptions import AppException
from app.mcp_servers import order_tool
from app.mcp_servers.minimal_server import mcp
from app.mcp_servers.order_tool import query_order_for_mcp
from tests.tool_fakes import FakeOrderLookupClient, make_java_order_payload


SENSITIVE_ORDER_VALUES = [
    "C_SECRET",
    "13800000000",
    "internal warehouse note",
]


def test_query_order_mcp_tool_schema_exposes_order_id_contract() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        tool = next(tool for tool in tools.tools if tool.name == "query_order")
        order_id_schema = tool.input_schema["properties"]["order_id"]
        assert tool.input_schema["required"] == ["order_id"]
        assert order_id_schema["minLength"] == 1
        assert order_id_schema["maxLength"] == 64
        assert order_id_schema["pattern"] == r"^[A-Za-z0-9_-]+$"

    asyncio.run(run())


def test_query_order_for_mcp_returns_safe_order_result() -> None:
    client = FakeOrderLookupClient(
        make_java_order_payload(
            customer_id="C_SECRET",
            customer_phone="13800000000",
            internal_note="internal warehouse note",
        )
    )

    response = query_order_for_mcp("A1001", client=client)

    assert response["ok"] is True
    assert response["allowed"] is True
    assert response["action_type"] == "read"
    assert response["requires_confirmation"] is False
    assert response["result"] == {
        "order_id": "A1001",
        "order_status": "waiting_shipment",
        "payment_status": "paid",
        "logistics_message": "商家已接单，等待仓库发货。",
        "latest_event": "仓库正在准备出库。",
        "can_create_ticket": True,
        "source": "java_business_service",
    }
    raw_response = json.dumps(response, ensure_ascii=False)
    for sensitive_value in SENSITIVE_ORDER_VALUES:
        assert sensitive_value not in raw_response
    assert client.calls == ["A1001"]


def test_query_order_for_mcp_returns_business_error_without_tool_error() -> None:
    client = FakeOrderLookupClient(
        error=AppException(
            code="ORDER_NOT_FOUND",
            message="订单不存在，请确认订单号是否正确。",
            status_code=404,
        )
    )

    response = query_order_for_mcp("A9999", client=client)

    assert response["ok"] is False
    assert response["allowed"] is True
    assert response["error_code"] == "ORDER_NOT_FOUND"
    assert response["message"] == "订单不存在，请确认订单号是否正确。"
    assert response["result"] is None


def test_query_order_for_mcp_returns_access_denied_business_error() -> None:
    client = FakeOrderLookupClient(
        error=AppException(
            code="ORDER_ACCESS_DENIED",
            message="当前账号无权查看或操作该订单。",
            status_code=403,
        )
    )

    response = query_order_for_mcp("A2001", client=client)

    assert response["ok"] is False
    assert response["error_code"] == "ORDER_ACCESS_DENIED"
    assert response["message"] == "当前账号无权查看或操作该订单。"


def test_query_order_for_mcp_returns_invalid_arguments_safely() -> None:
    response = query_order_for_mcp("A 1001", client=FakeOrderLookupClient())

    assert response["ok"] is False
    assert response["allowed"] is False
    assert response["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert response["result"] is None
    assert response["errors"][0]["loc"] == ("order_id",)


def test_query_order_for_mcp_raises_safe_tool_error_for_timeout() -> None:
    client = FakeOrderLookupClient(
        error=AppException(
            code="TOOL_TIMEOUT",
            message="内部 read timeout: http://java-business-service",
            status_code=504,
        )
    )

    with pytest.raises(ToolError) as exc_info:
        query_order_for_mcp("A_TIMEOUT", client=client)

    assert "TOOL_TIMEOUT" in str(exc_info.value)
    assert "java-business-service" not in str(exc_info.value)


def test_query_order_for_mcp_raises_safe_tool_error_for_upstream_failure() -> None:
    client = FakeOrderLookupClient(
        error=AppException(
            code="TOOL_RESULT_VALIDATION_FAILED",
            message="字段 customer_id_card 不符合内部契约。",
            status_code=502,
            details=[{"loc": ("customer_id_card",), "msg": "invalid"}],
        )
    )

    with pytest.raises(ToolError) as exc_info:
        query_order_for_mcp("A1001", client=client)

    error_text = str(exc_info.value)
    assert "TOOL_RESULT_VALIDATION_FAILED" in error_text
    assert "customer_id_card" not in error_text
    assert "内部契约" not in error_text


def test_query_order_mcp_client_call_can_use_fake_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_query_order_for_mcp(order_id: str) -> dict[str, object]:
        return {
            "ok": True,
            "allowed": True,
            "action": "query_order",
            "action_type": "read",
            "requires_confirmation": False,
            "error_code": None,
            "message": "订单查询成功。",
            "retryable": False,
            "security_checks": {
                "input_validated": True,
                "output_allowlist_applied": True,
                "sensitive_fields_returned": False,
            },
            "result": {"order_id": order_id},
        }

    async def run() -> None:
        monkeypatch.setattr(order_tool, "query_order_for_mcp", fake_query_order_for_mcp)
        async with Client(mcp) as client:
            result = await client.call_tool("query_order", {"order_id": "A1001"})

        assert result.is_error is False
        assert result.structured_content["ok"] is True
        assert result.structured_content["result"] == {"order_id": "A1001"}

    asyncio.run(run())
