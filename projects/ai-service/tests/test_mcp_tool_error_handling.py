import asyncio

from mcp import Client

from app.mcp_servers.minimal_server import mcp


def test_simulate_tool_error_handling_schema_exposes_scenarios() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        tool = next(
            tool for tool in tools.tools if tool.name == "simulate_tool_error_handling"
        )
        assert tool.input_schema["properties"]["scenario"]["enum"] == [
            "success",
            "business_not_found",
            "permission_denied",
            "upstream_timeout",
            "unexpected_failure",
        ]

    asyncio.run(run())


def test_business_not_found_returns_ok_false_without_tool_error() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "simulate_tool_error_handling",
                {"scenario": "business_not_found"},
            )

        assert result.is_error is False
        assert result.structured_content == {
            "ok": False,
            "error_code": "ORDER_NOT_FOUND",
            "message": "没有找到符合条件的订单，请确认订单号是否正确。",
            "retryable": False,
            "details": {"safe_reason": "order_not_found"},
        }

    asyncio.run(run())


def test_permission_denied_returns_safe_business_error() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "simulate_tool_error_handling",
                {"scenario": "permission_denied"},
            )

        assert result.is_error is False
        assert result.structured_content["ok"] is False
        assert result.structured_content["error_code"] == "ORDER_ACCESS_DENIED"
        assert "无权" in result.structured_content["message"]

    asyncio.run(run())


def test_upstream_timeout_returns_tool_error_with_safe_message() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "simulate_tool_error_handling",
                {"scenario": "upstream_timeout"},
            )

        assert result.is_error is True
        assert result.structured_content is None
        assert "UPSTREAM_TIMEOUT" in result.content[0].text
        assert "订单服务暂时没有响应" in result.content[0].text

    asyncio.run(run())


def test_unexpected_failure_is_wrapped_in_safe_tool_error() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "simulate_tool_error_handling",
                {"scenario": "unexpected_failure"},
            )

        assert result.is_error is True
        assert result.structured_content is None
        assert "INTERNAL_TOOL_ERROR" in result.content[0].text
        assert "simulated internal dependency failure" not in result.content[0].text

    asyncio.run(run())
