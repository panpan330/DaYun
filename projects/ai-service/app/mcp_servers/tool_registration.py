"""Tool registration for the learning MCP server."""

from typing import Any

from mcp.server import MCPServer

from app.mcp_servers import order_tool, ticket_tool
from app.mcp_servers.observability import observe_mcp_tool
from app.mcp_servers.ticket_validation import (
    TicketCategory,
    TicketDescription,
    TicketPriority,
    TicketTitle,
    validate_ticket_draft_arguments,
)
from app.mcp_servers.tool_error_handling import (
    ToolErrorScenario,
    simulate_tool_error_response,
)
from app.mcp_servers.tool_security import (
    SecurityScenario,
    build_tool_security_decision,
)


def echo(message: str) -> str:
    """Return the same message back to the caller."""
    return message


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


def validate_ticket_draft(
    title: TicketTitle,
    description: TicketDescription,
    category: TicketCategory,
    priority: TicketPriority = "normal",
) -> dict[str, Any]:
    """Validate a support ticket draft without creating a real ticket."""
    return validate_ticket_draft_arguments(
        title=title,
        description=description,
        category=category,
        priority=priority,
    )


def simulate_tool_error_handling(scenario: ToolErrorScenario) -> dict[str, Any]:
    """Simulate safe MCP tool success, business errors, and system errors."""
    return simulate_tool_error_response(scenario)


def inspect_tool_security_boundary(
    scenario: SecurityScenario,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Inspect MCP tool safety decisions without executing real business writes."""
    return build_tool_security_decision(
        scenario=scenario,
        user_confirmed=user_confirmed,
    )


def query_order(order_id: order_tool.OrderId) -> dict[str, Any]:
    """Query a business order through the MCP learning adapter."""
    return order_tool.query_order_for_mcp(order_id)


def create_ticket(
    requester_id: ticket_tool.RequesterId,
    title: ticket_tool.TicketTitle,
    description: ticket_tool.TicketDescription,
    category: ticket_tool.TicketCategory,
    confirmation_id: ticket_tool.ConfirmationId,
    priority: ticket_tool.TicketPriority = ticket_tool.TicketPriority.NORMAL,
    related_order_id: ticket_tool.RelatedOrderId = None,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Create a support ticket after user confirmation and idempotency checks."""
    return ticket_tool.create_ticket_for_mcp(
        requester_id=requester_id,
        title=title,
        description=description,
        category=category,
        priority=priority,
        related_order_id=related_order_id,
        confirmation_id=confirmation_id,
        user_confirmed=user_confirmed,
    )


def register_learning_tools(server: MCPServer) -> None:
    server.tool()(observe_mcp_tool(tool_name="echo", action_type="demo")(echo))
    server.tool()(observe_mcp_tool(tool_name="add", action_type="demo")(add))


def register_safety_learning_tools(server: MCPServer) -> None:
    server.tool()(
        observe_mcp_tool(tool_name="validate_ticket_draft", action_type="validation")(
            validate_ticket_draft
        )
    )
    server.tool()(
        observe_mcp_tool(
            tool_name="simulate_tool_error_handling",
            action_type="diagnostic",
        )(simulate_tool_error_handling)
    )
    server.tool()(
        observe_mcp_tool(
            tool_name="inspect_tool_security_boundary",
            action_type="diagnostic",
        )(inspect_tool_security_boundary)
    )


def register_business_tools(server: MCPServer) -> None:
    server.tool()(observe_mcp_tool(tool_name="query_order", action_type="read")(query_order))
    server.tool()(
        observe_mcp_tool(tool_name="create_ticket", action_type="write")(create_ticket)
    )


def register_all_tools(server: MCPServer) -> None:
    register_learning_tools(server)
    register_safety_learning_tools(server)
    register_business_tools(server)
