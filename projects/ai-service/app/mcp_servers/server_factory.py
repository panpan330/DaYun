"""Factory for assembling the learning MCP server."""

from mcp.server import MCPServer

from app.core.config import Settings, get_settings
from app.mcp_servers.resource_registration import register_all_resources
from app.mcp_servers.tool_registration import register_all_tools


def create_learning_mcp_server(
    settings: Settings | None = None,
    *,
    name: str | None = None,
) -> MCPServer:
    """Create a fully registered MCP server for the learning project."""
    resolved_settings = settings or get_settings()
    server = MCPServer(name or resolved_settings.resolved_mcp_server_name)
    register_all_tools(server)
    register_all_resources(
        server,
        include_learning_resources=resolved_settings.mcp_enable_learning_resources,
        include_project_resources=resolved_settings.mcp_enable_project_resources,
        project_resource_root=resolved_settings.resolved_mcp_project_resource_root,
    )
    return server
