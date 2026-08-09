import asyncio

from mcp import Client

from app.mcp_servers.minimal_server import mcp


def test_validate_ticket_draft_schema_exposes_required_fields_and_enums() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools = await client.list_tools()

        tool = next(tool for tool in tools.tools if tool.name == "validate_ticket_draft")
        schema = tool.input_schema
        properties = schema["properties"]

        assert set(schema["required"]) == {"title", "description", "category"}
        assert properties["title"]["minLength"] == 5
        assert properties["title"]["maxLength"] == 80
        assert properties["description"]["minLength"] == 10
        assert properties["description"]["maxLength"] == 500
        assert properties["category"]["enum"] == [
            "refund",
            "logistics",
            "order_issue",
            "other",
        ]
        assert properties["priority"]["enum"] == ["low", "normal", "high"]

    asyncio.run(run())


def test_validate_ticket_draft_accepts_valid_arguments() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "validate_ticket_draft",
                {
                    "title": "Refund request",
                    "description": "Customer asks about refund progress.",
                    "category": "refund",
                    "priority": "high",
                },
            )

        assert result.is_error is False
        assert result.structured_content == {
            "ok": True,
            "error_code": None,
            "errors": [],
            "draft": {
                "title": "Refund request",
                "description": "Customer asks about refund progress.",
                "category": "refund",
                "priority": "high",
            },
        }

    asyncio.run(run())


def test_validate_ticket_draft_returns_safe_business_validation_error() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "validate_ticket_draft",
                {
                    "title": "     ",
                    "description": "Description is long enough.",
                    "category": "other",
                    "priority": "normal",
                },
            )

        assert result.is_error is False
        assert result.structured_content["ok"] is False
        assert result.structured_content["error_code"] == "INVALID_TOOL_ARGUMENTS"
        assert result.structured_content["draft"] is None
        assert result.structured_content["errors"][0]["field"] == "title"

    asyncio.run(run())


def test_validate_ticket_draft_rejects_invalid_enum_at_schema_layer() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "validate_ticket_draft",
                {
                    "title": "Refund request",
                    "description": "Customer asks about refund progress.",
                    "category": "refund",
                    "priority": "urgent",
                },
            )

        assert result.is_error is True
        assert result.structured_content is None
        assert "priority" in result.content[0].text

    asyncio.run(run())
