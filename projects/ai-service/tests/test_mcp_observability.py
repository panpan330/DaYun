import asyncio
import logging

from mcp import Client

from app.core.trace import reset_trace_id, set_trace_id
from app.mcp_servers.minimal_server import mcp


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def test_mcp_tool_call_logs_safe_lifecycle_fields(caplog) -> None:
    async def run() -> None:
        trace_token = set_trace_id("trace-mcp-tool-001")
        try:
            with caplog.at_level(logging.INFO, logger="app.mcp"):
                async with Client(mcp) as client:
                    result = await client.call_tool("add", {"a": 7, "b": 5})
        finally:
            reset_trace_id(trace_token)

        assert result.structured_content == {"result": 12}

    asyncio.run(run())

    messages = _messages(caplog)
    joined_messages = "\n".join(messages)
    assert any("mcp_tool_call_started" in message for message in messages)
    assert any("mcp_tool_call_finished" in message for message in messages)
    assert "trace_id=trace-mcp-tool-001" in joined_messages
    assert "tool_name=add" in joined_messages
    assert "action_type=demo" in joined_messages
    assert "status=succeeded" in joined_messages
    assert "elapsed_ms=" in joined_messages
    assert "a=7" not in joined_messages
    assert "b=5" not in joined_messages


def test_mcp_tool_business_error_logs_error_code_without_payload(caplog) -> None:
    async def run() -> None:
        trace_token = set_trace_id("trace-mcp-tool-business-001")
        try:
            with caplog.at_level(logging.INFO, logger="app.mcp"):
                async with Client(mcp) as client:
                    result = await client.call_tool(
                        "create_ticket",
                        {
                            "requester_id": "demo_user_001",
                            "title": "订单 A1001 一直未发货",
                            "description": "订单 A1001 已付款一周仍未发货，请帮我处理。",
                            "category": "complaint",
                            "priority": "high",
                            "related_order_id": "A1001",
                            "confirmation_id": "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11",
                            "user_confirmed": False,
                        },
                    )
        finally:
            reset_trace_id(trace_token)

        assert result.is_error is False
        assert result.structured_content["error_code"] == "TOOL_CONFIRMATION_REQUIRED"

    asyncio.run(run())

    joined_messages = "\n".join(_messages(caplog))
    assert "tool_name=create_ticket" in joined_messages
    assert "action_type=write" in joined_messages
    assert "status=business_error" in joined_messages
    assert "error_code=TOOL_CONFIRMATION_REQUIRED" in joined_messages
    assert "订单 A1001 一直未发货" not in joined_messages
    assert "demo_user_001" not in joined_messages


def test_mcp_tool_system_error_logs_error_type_without_internal_message(caplog) -> None:
    async def run() -> None:
        trace_token = set_trace_id("trace-mcp-tool-system-001")
        try:
            with caplog.at_level(logging.INFO, logger="app.mcp"):
                async with Client(mcp) as client:
                    result = await client.call_tool(
                        "simulate_tool_error_handling",
                        {"scenario": "upstream_timeout"},
                    )
        finally:
            reset_trace_id(trace_token)

        assert result.is_error is True

    asyncio.run(run())

    joined_messages = "\n".join(_messages(caplog))
    assert "mcp_tool_call_failed" in joined_messages
    assert "tool_name=simulate_tool_error_handling" in joined_messages
    assert "status=system_error" in joined_messages
    assert "error_type=ToolError" in joined_messages
    assert "database_password" not in joined_messages
    assert "java-business-service" not in joined_messages


def test_mcp_resource_read_logs_safe_lifecycle_fields(caplog) -> None:
    async def run() -> None:
        trace_token = set_trace_id("trace-mcp-resource-001")
        try:
            with caplog.at_level(logging.INFO, logger="app.mcp"):
                async with Client(mcp) as client:
                    result = await client.read_resource("learning://project/readme")
        finally:
            reset_trace_id(trace_token)

        assert result.contents[0].mime_type == "text/markdown"

    asyncio.run(run())

    joined_messages = "\n".join(_messages(caplog))
    assert "mcp_resource_read_started" in joined_messages
    assert "mcp_resource_read_finished" in joined_messages
    assert "trace_id=trace-mcp-resource-001" in joined_messages
    assert "resource_uri=learning://project/readme" in joined_messages
    assert "mime_type=text/markdown" in joined_messages
    assert "status=succeeded" in joined_messages
    assert "阶段 8" not in joined_messages
    assert "MCP 与 AI 工具生态基础" not in joined_messages
