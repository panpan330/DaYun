"""Debug helpers for the minimal learning MCP server."""

from typing import Any

from mcp import Client

from app.mcp_servers.minimal_server import mcp


def _extract_text_items(items: list[Any]) -> list[str]:
    return [item.text for item in items if getattr(item, "text", None) is not None]


async def collect_minimal_mcp_debug_snapshot() -> dict[str, Any]:
    """Call the minimal MCP server and return a JSON-friendly debug snapshot."""
    async with Client(mcp) as client:
        tools_response = await client.list_tools()
        resources_response = await client.list_resources()
        resource_templates_response = await client.list_resource_templates()
        add_result = await client.call_tool("add", {"a": 7, "b": 5})
        echo_result = await client.call_tool("echo", {"message": "hello mcp"})
        ticket_validation_result = await client.call_tool(
            "validate_ticket_draft",
            {
                "title": "Logistics delay",
                "description": "A1001 logistics has not updated for three days.",
                "category": "logistics",
                "priority": "normal",
            },
        )
        business_error_result = await client.call_tool(
            "simulate_tool_error_handling",
            {"scenario": "business_not_found"},
        )
        system_error_result = await client.call_tool(
            "simulate_tool_error_handling",
            {"scenario": "upstream_timeout"},
        )
        sensitive_security_result = await client.call_tool(
            "inspect_tool_security_boundary",
            {"scenario": "sensitive_output_request"},
        )
        write_blocked_security_result = await client.call_tool(
            "inspect_tool_security_boundary",
            {"scenario": "write_without_confirmation"},
        )
        resource_result = await client.read_resource("learning://hello/panpan")
        java_ai_contract_result = await client.read_resource("learning://project/java-ai-contract")

    return {
        "server": "ai-service-learning-mcp",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
            for tool in tools_response.tools
        ],
        "resources": [
            {
                "uri": str(resource.uri),
                "name": resource.name,
                "title": resource.title,
                "description": resource.description,
                "mime_type": resource.mime_type,
            }
            for resource in resources_response.resources
        ],
        "resource_templates": [
            {
                "uri_template": template.uri_template,
                "name": template.name,
                "title": template.title,
                "description": template.description,
                "mime_type": template.mime_type,
            }
            for template in resource_templates_response.resource_templates
        ],
        "tool_calls": {
            "add": {
                "is_error": add_result.is_error,
                "structured_content": add_result.structured_content,
                "text_content": _extract_text_items(add_result.content),
            },
            "echo": {
                "is_error": echo_result.is_error,
                "structured_content": echo_result.structured_content,
                "text_content": _extract_text_items(echo_result.content),
            },
            "validate_ticket_draft": {
                "is_error": ticket_validation_result.is_error,
                "structured_content": ticket_validation_result.structured_content,
                "text_content": _extract_text_items(ticket_validation_result.content),
            },
            "simulate_tool_error_handling_business": {
                "is_error": business_error_result.is_error,
                "structured_content": business_error_result.structured_content,
                "text_content": _extract_text_items(business_error_result.content),
            },
            "simulate_tool_error_handling_system": {
                "is_error": system_error_result.is_error,
                "structured_content": system_error_result.structured_content,
                "text_content": _extract_text_items(system_error_result.content),
            },
            "inspect_tool_security_boundary_sensitive": {
                "is_error": sensitive_security_result.is_error,
                "structured_content": sensitive_security_result.structured_content,
                "text_content": _extract_text_items(sensitive_security_result.content),
            },
            "inspect_tool_security_boundary_write_blocked": {
                "is_error": write_blocked_security_result.is_error,
                "structured_content": write_blocked_security_result.structured_content,
                "text_content": _extract_text_items(write_blocked_security_result.content),
            },
        },
        "resource_reads": {
            "learning://hello/panpan": _extract_text_items(resource_result.contents),
            "learning://project/java-ai-contract": _extract_text_items(
                java_ai_contract_result.contents
            ),
        },
    }
