import asyncio

from app.mcp_clients.minimal_client import collect_minimal_mcp_debug_snapshot


def test_minimal_mcp_client_debug_snapshot() -> None:
    async def run() -> None:
        snapshot = await collect_minimal_mcp_debug_snapshot()

        tool_names = {tool["name"] for tool in snapshot["tools"]}
        assert {
            "add",
            "echo",
            "validate_ticket_draft",
            "simulate_tool_error_handling",
            "inspect_tool_security_boundary",
            "query_order",
            "create_ticket",
        }.issubset(tool_names)
        resource_uris = {resource["uri"] for resource in snapshot["resources"]}
        assert {
            "learning://project/readme",
            "learning://project/java-ai-contract",
        }.issubset(resource_uris)
        template_uris = {
            template["uri_template"] for template in snapshot["resource_templates"]
        }
        assert "learning://hello/{name}" in template_uris
        assert snapshot["tool_calls"]["add"]["is_error"] is False
        assert snapshot["tool_calls"]["add"]["structured_content"] == {"result": 12}
        assert snapshot["tool_calls"]["echo"]["structured_content"] == {
            "result": "hello mcp"
        }
        ticket_validation = snapshot["tool_calls"]["validate_ticket_draft"]
        assert ticket_validation["is_error"] is False
        assert ticket_validation["structured_content"]["ok"] is True
        assert ticket_validation["structured_content"]["draft"]["category"] == "logistics"
        business_error = snapshot["tool_calls"][
            "simulate_tool_error_handling_business"
        ]
        assert business_error["is_error"] is False
        assert business_error["structured_content"]["ok"] is False
        assert business_error["structured_content"]["error_code"] == "ORDER_NOT_FOUND"
        system_error = snapshot["tool_calls"]["simulate_tool_error_handling_system"]
        assert system_error["is_error"] is True
        assert system_error["structured_content"] is None
        assert "UPSTREAM_TIMEOUT" in system_error["text_content"][0]
        sensitive_security = snapshot["tool_calls"][
            "inspect_tool_security_boundary_sensitive"
        ]
        assert sensitive_security["is_error"] is False
        assert sensitive_security["structured_content"]["allowed"] is True
        assert (
            sensitive_security["structured_content"]["security_checks"][
                "blocked_field_count"
            ]
            == 3
        )
        assert "customer_phone" not in sensitive_security["structured_content"][
            "sanitized_output"
        ]["order"]
        write_blocked = snapshot["tool_calls"][
            "inspect_tool_security_boundary_write_blocked"
        ]
        assert write_blocked["is_error"] is False
        assert write_blocked["structured_content"]["allowed"] is False
        assert (
            write_blocked["structured_content"]["error_code"]
            == "USER_CONFIRMATION_REQUIRED"
        )
        assert snapshot["resource_reads"]["learning://hello/panpan"] == [
            "Hello, panpan. This resource comes from ai-service minimal MCP server."
        ]
    asyncio.run(run())
