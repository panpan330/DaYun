"""Compatibility entry point for the learning MCP server."""

from app.mcp_servers.server_factory import create_learning_mcp_server


mcp = create_learning_mcp_server()


if __name__ == "__main__":
    mcp.run()
