import asyncio

from mcp import Client

from app.mcp_servers.minimal_server import mcp


EXPECTED_TOOL_NAMES = {
    "echo",
    "add",
    "validate_ticket_draft",
    "simulate_tool_error_handling",
    "inspect_tool_security_boundary",
    "query_order",
    "create_ticket",
}

EXPECTED_RESOURCE_CONTRACTS = {
    "learning://project/readme": {
        "name": "project_readme_resource",
        "title": "Project README",
    },
    "learning://project/java-ai-contract": {
        "name": "java_ai_contract_resource",
        "title": "Java AI API Contract",
    },
}


def test_mcp_public_tool_names_are_stable() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools_response = await client.list_tools()

        tool_names = {tool.name for tool in tools_response.tools}

        assert tool_names == EXPECTED_TOOL_NAMES

    asyncio.run(run())


def test_query_order_tool_contract_is_stable() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools_response = await client.list_tools()

        tool = next(tool for tool in tools_response.tools if tool.name == "query_order")
        schema = tool.input_schema
        order_id = schema["properties"]["order_id"]

        assert schema["required"] == ["order_id"]
        assert order_id["type"] == "string"
        assert order_id["minLength"] == 1
        assert order_id["maxLength"] == 64
        assert order_id["pattern"] == r"^[A-Za-z0-9_-]+$"
        assert tool.output_schema["type"] == "object"

    asyncio.run(run())


def test_create_ticket_tool_contract_is_stable() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            tools_response = await client.list_tools()

        tool = next(tool for tool in tools_response.tools if tool.name == "create_ticket")
        schema = tool.input_schema
        properties = schema["properties"]

        assert schema["required"] == [
            "requester_id",
            "title",
            "description",
            "category",
            "confirmation_id",
        ]
        assert properties["requester_id"]["pattern"] == r"^[A-Za-z0-9_-]+$"
        assert properties["title"]["maxLength"] == 200
        assert properties["description"]["maxLength"] == 1000
        assert properties["confirmation_id"]["pattern"] == r"^[a-f0-9]{32}$"
        assert properties["priority"]["default"] == "normal"
        assert properties["user_confirmed"]["default"] is False
        assert schema["$defs"]["TicketCategory"]["enum"] == [
            "refund",
            "order_query",
            "logistics",
            "complaint",
            "policy_gap",
        ]
        assert schema["$defs"]["TicketPriority"]["enum"] == [
            "low",
            "normal",
            "high",
        ]
        assert tool.output_schema["type"] == "object"

    asyncio.run(run())


def test_create_ticket_write_error_contract_is_stable() -> None:
    async def run() -> None:
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

        assert result.is_error is False
        assert result.structured_content == {
            "ok": False,
            "allowed": False,
            "action": "create_ticket",
            "action_type": "write",
            "requires_confirmation": True,
            "confirmation_checked": False,
            "confirmation_id": "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11",
            "error_code": "TOOL_CONFIRMATION_REQUIRED",
            "message": "创建工单是写操作，必须先拿到用户确认，本次请求不会执行。",
            "retryable": False,
            "security_checks": {
                "input_validated": True,
                "user_confirmed": False,
                "idempotency_key_checked": True,
                "idempotency_key_source": "confirmation_id",
                "output_allowlist_applied": False,
                "sensitive_fields_returned": False,
            },
            "ticket": None,
        }

    asyncio.run(run())


def test_project_resource_contracts_are_stable() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            resources_response = await client.list_resources()

        resource_contracts = {
            str(resource.uri): {
                "name": resource.name,
                "title": resource.title,
                "mime_type": resource.mime_type,
            }
            for resource in resources_response.resources
        }

        assert set(resource_contracts) == set(EXPECTED_RESOURCE_CONTRACTS)
        for uri, expected in EXPECTED_RESOURCE_CONTRACTS.items():
            assert resource_contracts[uri] == {
                "name": expected["name"],
                "title": expected["title"],
                "mime_type": "text/markdown",
            }

    asyncio.run(run())


def test_project_resource_read_contract_is_stable() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.read_resource("learning://project/java-ai-contract")

        assert len(result.contents) == 1
        assert result.contents[0].uri == "learning://project/java-ai-contract"
        assert result.contents[0].mime_type == "text/markdown"
        assert "Java" in result.contents[0].text

    asyncio.run(run())
