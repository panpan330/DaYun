"""Resource registration for the learning MCP server."""

from pathlib import Path
from typing import Callable

from mcp.server import MCPServer

from app.mcp_servers.observability import observe_mcp_resource
from app.mcp_servers.project_resources import read_project_resource


PROJECT_RESOURCE_MIME_TYPE = "text/markdown"


def hello_resource(name: str) -> str:
    """Return a greeting resource for a learner."""
    return f"Hello, {name}. This resource comes from ai-service minimal MCP server."


def project_readme_resource() -> str:
    """Return the project README as MCP context."""
    return read_project_resource("learning://project/readme")


def java_ai_contract_resource() -> str:
    """Return the Java and AI API contract document as MCP context."""
    return read_project_resource("learning://project/java-ai-contract")




def build_project_resource_reader(
    uri: str,
    *,
    function_name: str,
    project_resource_root: Path | None,
) -> Callable[[], str]:
    def read_resource() -> str:
        return read_project_resource(uri, repo_root=project_resource_root)

    read_resource.__name__ = function_name
    return read_resource


def register_learning_resources(server: MCPServer) -> None:
    server.resource("learning://hello/{name}")(
        observe_mcp_resource(
            resource_uri="learning://hello/{name}",
            mime_type="text/plain",
        )(hello_resource)
    )


def register_project_resources(
    server: MCPServer,
    *,
    project_resource_root: Path | None = None,
) -> None:
    server.resource(
        "learning://project/readme",
        title="Project README",
        description="GitHub homepage and project learning entry.",
        mime_type=PROJECT_RESOURCE_MIME_TYPE,
    )(
        observe_mcp_resource(
            resource_uri="learning://project/readme",
            mime_type=PROJECT_RESOURCE_MIME_TYPE,
        )(
            build_project_resource_reader(
                "learning://project/readme",
                function_name="project_readme_resource",
                project_resource_root=project_resource_root,
            )
        )
    )
    server.resource(
        "learning://project/java-ai-contract",
        title="Java AI API Contract",
        description="Contract between Python AI service and Java business service.",
        mime_type=PROJECT_RESOURCE_MIME_TYPE,
    )(
        observe_mcp_resource(
            resource_uri="learning://project/java-ai-contract",
            mime_type=PROJECT_RESOURCE_MIME_TYPE,
        )(
            build_project_resource_reader(
                "learning://project/java-ai-contract",
                function_name="java_ai_contract_resource",
                project_resource_root=project_resource_root,
            )
        )
    )
def register_all_resources(
    server: MCPServer,
    *,
    include_learning_resources: bool = True,
    include_project_resources: bool = True,
    project_resource_root: Path | None = None,
) -> None:
    if include_learning_resources:
        register_learning_resources(server)
    if include_project_resources:
        register_project_resources(
            server,
            project_resource_root=project_resource_root,
        )
