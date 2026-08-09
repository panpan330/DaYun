from langchain_core.tools import StructuredTool

from app.core.config import Settings
from app.schemas.tool import QueryOrderArgs
from app.tools.langchain_tools import (
    create_query_order_langchain_tool,
    get_langchain_tool_metadata,
    list_model_callable_langchain_tools,
)
from tests.tool_fakes import FakeOrderLookupClient, make_java_order_payload


def test_create_query_order_langchain_tool_exposes_backend_definition() -> None:
    tool = create_query_order_langchain_tool(
        settings=Settings(_env_file=None),
        client=FakeOrderLookupClient(),
    )

    assert isinstance(tool, StructuredTool)
    assert tool.name == "query_order"
    assert tool.description == "查询订单状态和物流摘要，只读取订单信息，不修改业务数据。"
    assert tool.args_schema is QueryOrderArgs
    assert "order_id" in tool.args


def test_query_order_langchain_tool_invokes_existing_query_order_logic() -> None:
    fake_client = FakeOrderLookupClient(
        make_java_order_payload(order_id="A1002", order_status="shipped")
    )
    tool = create_query_order_langchain_tool(
        settings=Settings(_env_file=None),
        client=fake_client,
    )

    result = tool.invoke({"order_id": " A1002 "})

    assert result == {
        "order_id": "A1002",
        "order_status": "shipped",
        "payment_status": "paid",
        "logistics_message": "商家已接单，等待仓库发货。",
        "latest_event": "仓库正在准备出库。",
        "can_create_ticket": True,
        "source": "java_business_service",
    }
    assert fake_client.calls == ["A1002"]


def test_get_langchain_tool_metadata_returns_model_visible_shape() -> None:
    tool = create_query_order_langchain_tool(
        settings=Settings(_env_file=None),
        client=FakeOrderLookupClient(),
    )

    metadata = get_langchain_tool_metadata(tool)

    assert metadata["name"] == "query_order"
    assert metadata["description"] == "查询订单状态和物流摘要，只读取订单信息，不修改业务数据。"
    assert metadata["args_schema"] == tool.args


def test_list_model_callable_langchain_tools_only_exposes_read_tool() -> None:
    tools = list_model_callable_langchain_tools(
        settings=Settings(_env_file=None),
        clients={"query_order": FakeOrderLookupClient()},
    )

    assert [tool.name for tool in tools] == ["query_order"]
