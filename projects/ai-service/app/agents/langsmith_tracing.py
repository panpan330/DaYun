from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.agents.thread_lifecycle import normalize_ticket_agent_thread_id
from app.core.trace import get_trace_id


LangSmithMetadataValue = str | int | float | bool

TICKET_AGENT_LANGSMITH_PROJECT_NAME = "ai-service-ticket-agent"
TICKET_AGENT_LANGSMITH_RUN_NAME_PREFIX = "ticket_agent"
LANGSMITH_TAG_MAX_LENGTH = 64
LANGSMITH_METADATA_KEY_MAX_LENGTH = 64
LANGSMITH_METADATA_TEXT_MAX_LENGTH = 200

TICKET_AGENT_LANGSMITH_BASE_TAGS = (
    "ai-service",
    "ticket-agent",
    "langgraph",
)
TICKET_AGENT_LANGSMITH_PROTECTED_METADATA_KEYS = frozenset(
    {
        "component",
        "operation",
        "trace_id",
        "thread_id",
        "session_id",
        "actor_id",
    }
)
TICKET_AGENT_LANGSMITH_SENSITIVE_METADATA_KEYS = frozenset(
    {
        "user_message",
        "normalized_message",
        "rag_query",
        "rag_answer",
        "rag_citations",
        "rag_suggestions",
        "final_answer",
        "ticket_fields",
        "ticket_creation_args",
        "created_ticket",
        "order_query_result",
        "pending_ticket_confirmation",
    }
)

_TAG_UNSAFE_PATTERN = re.compile(r"[^a-z0-9_.:-]+")
_METADATA_KEY_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.:-]+")


@dataclass(frozen=True)
class TicketAgentLangSmithTraceContext:
    project_name: str
    run_name: str
    tags: list[str]
    metadata: dict[str, LangSmithMetadataValue]
    thread_id: str | None = None

    def to_langgraph_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "run_name": self.run_name,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        if self.thread_id is not None:
            config["configurable"] = {"thread_id": self.thread_id}
        return config

    def to_tracing_context_kwargs(self, *, enabled: bool = True) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "enabled": enabled,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def normalize_langsmith_tag(tag: object) -> str | None:
    if tag is None:
        return None
    text = str(tag).strip().casefold()
    if not text:
        return None

    normalized_tag = _TAG_UNSAFE_PATTERN.sub("-", text).strip("-._:")
    if not normalized_tag:
        return None
    if len(normalized_tag) > LANGSMITH_TAG_MAX_LENGTH:
        return normalized_tag[:LANGSMITH_TAG_MAX_LENGTH].rstrip("-._:")
    return normalized_tag


def build_langsmith_trace_tags(
    *,
    environment: str = "local",
    operation: str | None = None,
    intent: str | None = None,
    extra_tags: Iterable[object] | None = None,
) -> list[str]:
    tags: list[object] = [
        *TICKET_AGENT_LANGSMITH_BASE_TAGS,
        f"env:{environment}",
    ]
    if operation is not None:
        tags.append(f"operation:{operation}")
    if intent is not None:
        tags.append(f"intent:{intent}")
    if extra_tags is not None:
        tags.extend(extra_tags)
    return _normalize_deduplicated_tags(tags)


def build_ticket_agent_langsmith_metadata(
    state: Mapping[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    actor_id: str | None = None,
    elapsed_ms: float | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, LangSmithMetadataValue]:
    metadata: dict[str, LangSmithMetadataValue] = {}

    _add_metadata_value(metadata, "component", "ticket_agent")
    _add_metadata_value(metadata, "operation", operation)
    _add_metadata_value(
        metadata,
        "trace_id",
        state.get("agent_trace_id") or get_trace_id(),
    )

    normalized_thread_id = _normalize_optional_thread_id(thread_id)
    if normalized_thread_id is not None:
        _add_metadata_value(metadata, "thread_id", normalized_thread_id)
        _add_metadata_value(metadata, "session_id", normalized_thread_id)

    selected_actor_id = actor_id or state.get("ticket_actor_id")
    _add_metadata_value(metadata, "actor_id", selected_actor_id)

    node_history = state.get("node_history")
    node_count, last_node = _summarize_node_history(node_history)
    _add_metadata_value(metadata, "node_count", node_count)
    _add_metadata_value(metadata, "last_node", last_node)

    for field_name in (
        "intent",
        "ticket_need_source",
        "order_query_status",
        "order_query_error_code",
        "order_query_error_kind",
        "order_query_error_action",
        "rag_answer_status",
        "rag_no_context_reason",
        "ticket_field_extraction_source",
        "ticket_fields_complete",
        "ticket_confirmation_required",
        "ticket_confirmation_approved",
        "ticket_tool_name",
        "ticket_tool_access_level",
        "ticket_tool_requires_confirmation",
        "ticket_write_safety_status",
        "ticket_creation_status",
        "ticket_creation_error_code",
        "agent_error_code",
        "agent_error_node",
        "fallback_used",
    ):
        _add_metadata_value(metadata, field_name, state.get(field_name))

    _add_metadata_value(
        metadata,
        "rag_citation_count",
        _count_collection_items(state.get("rag_citations")),
    )
    _add_metadata_value(
        metadata,
        "missing_ticket_fields_count",
        _count_collection_items(state.get("missing_ticket_fields")),
    )

    if elapsed_ms is not None and math.isfinite(elapsed_ms):
        metadata["elapsed_ms"] = round(elapsed_ms, 2)

    if extra_metadata is not None:
        _merge_extra_metadata(metadata, extra_metadata)

    return metadata


def build_ticket_agent_langsmith_trace_context(
    state: Mapping[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    actor_id: str | None = None,
    environment: str = "local",
    project_name: str = TICKET_AGENT_LANGSMITH_PROJECT_NAME,
    elapsed_ms: float | None = None,
    extra_tags: Iterable[object] | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> TicketAgentLangSmithTraceContext:
    metadata = build_ticket_agent_langsmith_metadata(
        state,
        operation=operation,
        thread_id=thread_id,
        actor_id=actor_id,
        elapsed_ms=elapsed_ms,
        extra_metadata=extra_metadata,
    )
    tags = build_langsmith_trace_tags(
        environment=environment,
        operation=operation,
        intent=_metadata_string(metadata.get("intent")),
        extra_tags=extra_tags,
    )
    normalized_operation = normalize_langsmith_tag(operation) or "unknown"
    normalized_thread_id = _metadata_string(metadata.get("thread_id"))
    return TicketAgentLangSmithTraceContext(
        project_name=project_name,
        run_name=f"{TICKET_AGENT_LANGSMITH_RUN_NAME_PREFIX}.{normalized_operation}",
        tags=tags,
        metadata=metadata,
        thread_id=normalized_thread_id,
    )


def _normalize_optional_thread_id(thread_id: str | None) -> str | None:
    if thread_id is None:
        return None
    return normalize_ticket_agent_thread_id(thread_id)


def _normalize_deduplicated_tags(tags: Iterable[object]) -> list[str]:
    normalized_tags: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized_tag = normalize_langsmith_tag(tag)
        if normalized_tag is None or normalized_tag in seen:
            continue
        normalized_tags.append(normalized_tag)
        seen.add(normalized_tag)
    return normalized_tags


def _normalize_metadata_key(key: object) -> str | None:
    if key is None:
        return None
    text = str(key).strip().replace(" ", "_")
    if not text:
        return None
    normalized_key = _METADATA_KEY_UNSAFE_PATTERN.sub("_", text).strip("_.:-")
    if not normalized_key:
        return None
    if len(normalized_key) > LANGSMITH_METADATA_KEY_MAX_LENGTH:
        return normalized_key[:LANGSMITH_METADATA_KEY_MAX_LENGTH].rstrip("_.:-")
    return normalized_key


def _safe_metadata_value(value: object) -> LangSmithMetadataValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return _trim_metadata_text(value)
    return None


def _trim_metadata_text(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if len(text) > LANGSMITH_METADATA_TEXT_MAX_LENGTH:
        return f"{text[: LANGSMITH_METADATA_TEXT_MAX_LENGTH - 3]}..."
    return text


def _add_metadata_value(
    metadata: dict[str, LangSmithMetadataValue],
    key: object,
    value: object,
) -> None:
    normalized_key = _normalize_metadata_key(key)
    if normalized_key is None:
        return
    if normalized_key in TICKET_AGENT_LANGSMITH_SENSITIVE_METADATA_KEYS:
        return
    safe_value = _safe_metadata_value(value)
    if safe_value is None:
        return
    metadata[normalized_key] = safe_value


def _merge_extra_metadata(
    metadata: dict[str, LangSmithMetadataValue],
    extra_metadata: Mapping[str, object],
) -> None:
    for key, value in extra_metadata.items():
        normalized_key = _normalize_metadata_key(key)
        if normalized_key is None:
            continue
        if normalized_key in TICKET_AGENT_LANGSMITH_PROTECTED_METADATA_KEYS:
            continue
        if normalized_key in TICKET_AGENT_LANGSMITH_SENSITIVE_METADATA_KEYS:
            continue
        if normalized_key in metadata:
            continue
        safe_value = _safe_metadata_value(value)
        if safe_value is None:
            continue
        metadata[normalized_key] = safe_value


def _summarize_node_history(node_history: object) -> tuple[int, str | None]:
    if not isinstance(node_history, (list, tuple)):
        return 0, None
    if not node_history:
        return 0, None
    return len(node_history), str(node_history[-1])


def _count_collection_items(value: object) -> int:
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return 0


def _metadata_string(value: LangSmithMetadataValue | None) -> str | None:
    if isinstance(value, str):
        return value
    return None
