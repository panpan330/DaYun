"""Allowlisted project document resources for MCP learning examples."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectResourceSpec:
    uri: str
    title: str
    description: str
    relative_path: str
    mime_type: str = "text/markdown"


PROJECT_RESOURCE_SPECS: dict[str, ProjectResourceSpec] = {
    "learning://project/readme": ProjectResourceSpec(
        uri="learning://project/readme",
        title="Project README",
        description="GitHub homepage and project learning entry.",
        relative_path="README.md",
    ),
    "learning://project/java-ai-contract": ProjectResourceSpec(
        uri="learning://project/java-ai-contract",
        title="Java AI API Contract",
        description="Contract between Python AI service and Java business service.",
        relative_path="docs/java-ai-api-contract.md",
    ),
}


def find_learning_repo_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "README.md").is_file() and (
            parent / "projects" / "ai-service"
        ).is_dir():
            return parent
    raise RuntimeError("Could not locate the learning repository root.")


def list_project_resource_specs() -> list[ProjectResourceSpec]:
    return list(PROJECT_RESOURCE_SPECS.values())


def get_project_resource_spec(uri: str) -> ProjectResourceSpec:
    try:
        return PROJECT_RESOURCE_SPECS[uri]
    except KeyError as exc:
        raise ValueError(f"Project resource is not allowlisted: {uri}") from exc


def read_project_resource(uri: str, *, repo_root: Path | None = None) -> str:
    spec = get_project_resource_spec(uri)
    resolved_repo_root = (repo_root or find_learning_repo_root()).resolve()
    resource_path = (resolved_repo_root / spec.relative_path).resolve()

    if (
        resolved_repo_root not in resource_path.parents
        and resource_path != resolved_repo_root
    ):
        raise ValueError("Project resource path escaped the repository root.")

    return resource_path.read_text(encoding="utf-8")
