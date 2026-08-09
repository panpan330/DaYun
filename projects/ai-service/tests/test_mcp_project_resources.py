import asyncio
from pathlib import Path

import pytest
from mcp import Client

from app.mcp_servers.minimal_server import mcp
from app.mcp_servers.project_resources import (
    PROJECT_RESOURCE_SPECS,
    find_learning_repo_root,
    get_project_resource_spec,
    list_project_resource_specs,
    read_project_resource,
)


def test_project_resources_are_allowlisted_documents() -> None:
    specs = list_project_resource_specs()

    assert len(specs) == 2
    assert {spec.uri for spec in specs} == set(PROJECT_RESOURCE_SPECS)
    for spec in specs:
        path = Path(spec.relative_path)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert ".env" not in path.parts
        assert spec.mime_type == "text/markdown"


def test_read_project_resource_reads_only_known_resource() -> None:
    content = read_project_resource("learning://project/readme")

    assert "AI 客服" in content


def test_unknown_project_resource_is_rejected() -> None:
    with pytest.raises(ValueError, match="not allowlisted"):
        get_project_resource_spec("learning://project/.env")


def test_learning_repo_root_is_current_repository() -> None:
    repo_root = find_learning_repo_root()

    assert (repo_root / "README.md").is_file()
    assert (repo_root / "projects" / "ai-service").is_dir()


def test_mcp_client_lists_project_document_resources() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            resources = await client.list_resources()

        resource_uris = {str(resource.uri) for resource in resources.resources}
        assert "learning://project/readme" in resource_uris

    asyncio.run(run())


def test_mcp_client_reads_project_document_resource() -> None:
    async def run() -> None:
        async with Client(mcp) as client:
            result = await client.read_resource("learning://project/readme")

        assert result.contents[0].mime_type == "text/markdown"
        assert "AI 客服" in result.contents[0].text

    asyncio.run(run())
