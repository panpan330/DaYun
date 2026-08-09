from collections.abc import Mapping
from typing import Any

from langchain_core.tools import StructuredTool

from app.core.config import Settings
from app.schemas.tool import QueryOrderArgs
from app.tools.fake_order_tool import OrderLookupClient, query_order
from app.tools.tool_registry import authorize_tool_call, get_tool_definition


def create_query_order_langchain_tool(
    *,
    settings: Settings | None = None,
    client: OrderLookupClient | None = None,
) -> StructuredTool:
    definition = get_tool_definition("query_order")
    if definition is None:
        raise RuntimeError("query_order tool definition is missing")

    def _query_order(order_id: str) -> dict[str, Any]:
        authorize_tool_call("query_order")
        arguments = QueryOrderArgs(order_id=order_id)
        result = query_order(arguments, client=client, settings=settings)
        return result.model_dump(mode="json")

    return StructuredTool.from_function(
        _query_order,
        name=definition.name,
        description=definition.description,
        args_schema=QueryOrderArgs,
    )


def get_langchain_tool_metadata(tool: StructuredTool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "args_schema": tool.args,
    }


def list_model_callable_langchain_tools(
    *,
    settings: Settings | None = None,
    clients: Mapping[str, OrderLookupClient] | None = None,
) -> list[StructuredTool]:
    clients = clients or {}
    return [
        create_query_order_langchain_tool(
            settings=settings,
            client=clients.get("query_order"),
        )
    ]
