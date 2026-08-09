from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from secrets import token_hex
from typing import Any, Literal
from uuid import uuid4

from app.agents.thread_lifecycle import normalize_ticket_agent_thread_id
from app.core.trace import get_trace_id


OtelAttributeValue = str | int | float | bool
OtelSpanKind = Literal["INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER"]
OtelSpanStatus = Literal["UNSET", "OK", "ERROR"]

TRACEPARENT_HEADER = "traceparent"
TRACESTATE_HEADER = "tracestate"
OTEL_TRACEPARENT_VERSION = "00"
OTEL_TRACEPARENT_SAMPLED_FLAG = "01"
OTEL_TRACEPARENT_NOT_SAMPLED_FLAG = "00"
OTEL_TRACE_ID_HEX_LENGTH = 32
OTEL_SPAN_ID_HEX_LENGTH = 16

TICKET_AGENT_OTEL_SERVICE_NAME = "ai-service"
TICKET_AGENT_OTEL_SERVICE_NAMESPACE = "java-python-ai"
TICKET_AGENT_OTEL_SCOPE_NAME = "app.agents.ticket_agent"
TICKET_AGENT_OTEL_SPAN_NAME_PREFIX = "ticket_agent"

TICKET_AGENT_OTEL_PROTECTED_ATTRIBUTE_KEYS = frozenset(
    {
        "app.component",
        "app.operation",
        "app.trace_id",
        "app.thread_id",
        "app.session_id",
        "app.actor_id",
        "otel.scope.name",
        "service.name",
        "service.namespace",
        "service.version",
        "deployment.environment.name",
    }
)
TICKET_AGENT_OTEL_SENSITIVE_STATE_KEYS = frozenset(
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

_LOWER_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")
_TRACEPARENT_PATTERN = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-"
    r"(?P<trace_flags>[0-9a-f]{2})$"
)
_ATTRIBUTE_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ATTRIBUTE_KEY_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class OtelTraceParent:
    version: str
    trace_id: str
    parent_id: str
    trace_flags: str

    @property
    def sampled(self) -> bool:
        return int(self.trace_flags, 16) & 1 == 1

    def format(self) -> str:
        return (
            f"{self.version}-{self.trace_id}-{self.parent_id}-{self.trace_flags}"
        )


@dataclass(frozen=True)
class OtelTraceContext:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    sampled: bool = True

    def to_traceparent(self) -> str:
        return build_traceparent(
            trace_id=self.trace_id,
            span_id=self.span_id,
            sampled=self.sampled,
        )

    def to_headers(self) -> dict[str, str]:
        return {TRACEPARENT_HEADER: self.to_traceparent()}


@dataclass(frozen=True)
class TicketAgentOtelSpanPlan:
    span_name: str
    span_kind: OtelSpanKind
    status: OtelSpanStatus
    attributes: dict[str, OtelAttributeValue]
    trace_context: OtelTraceContext
    status_description: str | None = None

    def to_start_span_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.span_name,
            "kind": self.span_kind,
            "attributes": dict(self.attributes),
        }


def generate_otel_trace_id() -> str:
    return uuid4().hex


def generate_otel_span_id() -> str:
    span_id = token_hex(OTEL_SPAN_ID_HEX_LENGTH // 2)
    if _is_all_zero_hex(span_id):
        return generate_otel_span_id()
    return span_id


def normalize_otel_trace_id(trace_id: str | None) -> str | None:
    if trace_id is None:
        return None
    normalized_trace_id = trace_id.strip().casefold()
    if len(normalized_trace_id) != OTEL_TRACE_ID_HEX_LENGTH:
        return None
    if not _LOWER_HEX_PATTERN.fullmatch(normalized_trace_id):
        return None
    if _is_all_zero_hex(normalized_trace_id):
        return None
    return normalized_trace_id


def normalize_otel_span_id(span_id: str | None) -> str | None:
    if span_id is None:
        return None
    normalized_span_id = span_id.strip().casefold()
    if len(normalized_span_id) != OTEL_SPAN_ID_HEX_LENGTH:
        return None
    if not _LOWER_HEX_PATTERN.fullmatch(normalized_span_id):
        return None
    if _is_all_zero_hex(normalized_span_id):
        return None
    return normalized_span_id


def parse_traceparent(traceparent: str | None) -> OtelTraceParent | None:
    if traceparent is None:
        return None
    normalized_traceparent = traceparent.strip().casefold()
    match = _TRACEPARENT_PATTERN.fullmatch(normalized_traceparent)
    if match is None:
        return None

    version = match.group("version")
    trace_id = normalize_otel_trace_id(match.group("trace_id"))
    parent_id = normalize_otel_span_id(match.group("parent_id"))
    trace_flags = match.group("trace_flags")

    if version == "ff" or trace_id is None or parent_id is None:
        return None

    return OtelTraceParent(
        version=version,
        trace_id=trace_id,
        parent_id=parent_id,
        trace_flags=trace_flags,
    )


def build_traceparent(
    *,
    trace_id: str,
    span_id: str,
    sampled: bool = True,
) -> str:
    normalized_trace_id = normalize_otel_trace_id(trace_id)
    normalized_span_id = normalize_otel_span_id(span_id)
    if normalized_trace_id is None:
        raise ValueError("OpenTelemetry trace_id must be 32 non-zero hex chars.")
    if normalized_span_id is None:
        raise ValueError("OpenTelemetry span_id must be 16 non-zero hex chars.")

    trace_flags = (
        OTEL_TRACEPARENT_SAMPLED_FLAG
        if sampled
        else OTEL_TRACEPARENT_NOT_SAMPLED_FLAG
    )
    return (
        f"{OTEL_TRACEPARENT_VERSION}-"
        f"{normalized_trace_id}-"
        f"{normalized_span_id}-"
        f"{trace_flags}"
    )


def build_otel_trace_context(
    *,
    incoming_traceparent: str | None = None,
    project_trace_id: str | None = None,
    span_id: str | None = None,
    sampled: bool = True,
) -> OtelTraceContext:
    parent = parse_traceparent(incoming_traceparent)
    if parent is not None:
        selected_trace_id = parent.trace_id
        parent_span_id = parent.parent_id
        selected_sampled = parent.sampled
    else:
        selected_trace_id = (
            normalize_otel_trace_id(project_trace_id)
            or normalize_otel_trace_id(get_trace_id())
            or generate_otel_trace_id()
        )
        parent_span_id = None
        selected_sampled = sampled

    selected_span_id = normalize_otel_span_id(span_id) or generate_otel_span_id()
    return OtelTraceContext(
        trace_id=selected_trace_id,
        span_id=selected_span_id,
        parent_span_id=parent_span_id,
        sampled=selected_sampled,
    )


def build_ticket_agent_otel_resource_attributes(
    *,
    service_name: str = TICKET_AGENT_OTEL_SERVICE_NAME,
    service_namespace: str = TICKET_AGENT_OTEL_SERVICE_NAMESPACE,
    service_version: str | None = None,
    environment: str = "local",
) -> dict[str, OtelAttributeValue]:
    attributes: dict[str, OtelAttributeValue] = {}
    _add_otel_attribute(attributes, "service.name", service_name)
    _add_otel_attribute(attributes, "service.namespace", service_namespace)
    _add_otel_attribute(attributes, "deployment.environment.name", environment)
    _add_otel_attribute(attributes, "service.version", service_version)
    return attributes


def build_ticket_agent_otel_span_attributes(
    state: Mapping[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    actor_id: str | None = None,
    elapsed_ms: float | None = None,
    extra_attributes: Mapping[str, object] | None = None,
) -> dict[str, OtelAttributeValue]:
    attributes: dict[str, OtelAttributeValue] = {}

    _add_otel_attribute(attributes, "otel.scope.name", TICKET_AGENT_OTEL_SCOPE_NAME)
    _add_otel_attribute(attributes, "app.component", "ticket_agent")
    _add_otel_attribute(attributes, "app.operation", operation)
    _add_otel_attribute(
        attributes,
        "app.trace_id",
        state.get("agent_trace_id") or get_trace_id(),
    )

    normalized_thread_id = _normalize_optional_thread_id(thread_id)
    if normalized_thread_id is not None:
        _add_otel_attribute(attributes, "app.thread_id", normalized_thread_id)
        _add_otel_attribute(attributes, "app.session_id", normalized_thread_id)

    selected_actor_id = actor_id or state.get("ticket_actor_id")
    _add_otel_attribute(attributes, "app.actor_id", selected_actor_id)

    node_count, last_node = _summarize_node_history(state.get("node_history"))
    _add_otel_attribute(attributes, "agent.node.count", node_count)
    _add_otel_attribute(attributes, "agent.node.last", last_node)

    for state_key, attribute_key in (
        ("intent", "agent.intent"),
        ("ticket_need_source", "ticket.need.source"),
        ("order_query_status", "order.query.status"),
        ("order_query_error_code", "order.query.error_code"),
        ("order_query_error_kind", "order.query.error_kind"),
        ("order_query_error_action", "order.query.error_action"),
        ("rag_answer_status", "rag.answer.status"),
        ("rag_no_context_reason", "rag.no_context.reason"),
        ("ticket_field_extraction_source", "ticket.field_extraction.source"),
        ("ticket_fields_complete", "ticket.fields.complete"),
        ("ticket_confirmation_required", "ticket.confirmation.required"),
        ("ticket_confirmation_approved", "ticket.confirmation.approved"),
        ("ticket_tool_name", "ticket.tool.name"),
        ("ticket_tool_access_level", "ticket.tool.access_level"),
        ("ticket_tool_requires_confirmation", "ticket.tool.requires_confirmation"),
        ("ticket_write_safety_status", "ticket.write_safety.status"),
        ("ticket_creation_status", "ticket.creation.status"),
        ("ticket_creation_error_code", "ticket.creation.error_code"),
        ("agent_error_code", "agent.error_code"),
        ("agent_error_node", "agent.error_node"),
        ("fallback_used", "agent.fallback_used"),
    ):
        _add_otel_attribute(attributes, attribute_key, state.get(state_key))

    _add_otel_attribute(
        attributes,
        "rag.citation.count",
        _count_collection_items(state.get("rag_citations")),
    )
    _add_otel_attribute(
        attributes,
        "ticket.missing_fields.count",
        _count_collection_items(state.get("missing_ticket_fields")),
    )

    if elapsed_ms is not None and math.isfinite(elapsed_ms):
        attributes["app.elapsed_ms"] = round(elapsed_ms, 2)

    if extra_attributes is not None:
        _merge_extra_attributes(attributes, extra_attributes)

    return attributes


def build_ticket_agent_otel_span_plan(
    state: Mapping[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    actor_id: str | None = None,
    incoming_traceparent: str | None = None,
    span_id: str | None = None,
    span_kind: OtelSpanKind = "INTERNAL",
    elapsed_ms: float | None = None,
    extra_attributes: Mapping[str, object] | None = None,
) -> TicketAgentOtelSpanPlan:
    trace_context = build_otel_trace_context(
        incoming_traceparent=incoming_traceparent,
        project_trace_id=state.get("agent_trace_id"),
        span_id=span_id,
    )
    attributes = build_ticket_agent_otel_span_attributes(
        state,
        operation=operation,
        thread_id=thread_id,
        actor_id=actor_id,
        elapsed_ms=elapsed_ms,
        extra_attributes=extra_attributes,
    )
    status, status_description = _infer_ticket_agent_span_status(state)
    return TicketAgentOtelSpanPlan(
        span_name=_build_ticket_agent_span_name(operation),
        span_kind=span_kind,
        status=status,
        status_description=status_description,
        attributes=attributes,
        trace_context=trace_context,
    )


def _build_ticket_agent_span_name(operation: str) -> str:
    normalized_operation = operation.strip().casefold().replace(" ", "_")
    if not normalized_operation:
        normalized_operation = "unknown"
    return f"{TICKET_AGENT_OTEL_SPAN_NAME_PREFIX}.{normalized_operation}"


def _infer_ticket_agent_span_status(
    state: Mapping[str, Any],
) -> tuple[OtelSpanStatus, str | None]:
    agent_error_code = _safe_otel_attribute_value(state.get("agent_error_code"))
    if isinstance(agent_error_code, str):
        return "ERROR", agent_error_code

    ticket_creation_error_code = _safe_otel_attribute_value(
        state.get("ticket_creation_error_code")
    )
    if isinstance(ticket_creation_error_code, str):
        return "ERROR", ticket_creation_error_code

    if state.get("ticket_creation_status") == "failed":
        return "ERROR", "ticket_creation_failed"

    return "UNSET", None


def _normalize_optional_thread_id(thread_id: str | None) -> str | None:
    if thread_id is None:
        return None
    return normalize_ticket_agent_thread_id(thread_id)


def _normalize_otel_attribute_key(key: object) -> str | None:
    if key is None:
        return None
    text = str(key).strip().replace(" ", "_")
    if not text:
        return None
    normalized_key = (
        _ATTRIBUTE_KEY_UNSAFE_PATTERN.sub("_", text).strip("_.-").casefold()
    )
    if not normalized_key:
        return None
    if not _ATTRIBUTE_KEY_PATTERN.fullmatch(normalized_key):
        return None
    return normalized_key


def _safe_otel_attribute_value(value: object) -> OtelAttributeValue | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _add_otel_attribute(
    attributes: dict[str, OtelAttributeValue],
    key: object,
    value: object,
) -> None:
    normalized_key = _normalize_otel_attribute_key(key)
    if normalized_key is None:
        return
    if normalized_key in TICKET_AGENT_OTEL_SENSITIVE_STATE_KEYS:
        return
    safe_value = _safe_otel_attribute_value(value)
    if safe_value is None:
        return
    attributes[normalized_key] = safe_value


def _merge_extra_attributes(
    attributes: dict[str, OtelAttributeValue],
    extra_attributes: Mapping[str, object],
) -> None:
    for key, value in extra_attributes.items():
        normalized_key = _normalize_otel_attribute_key(key)
        if normalized_key is None:
            continue
        if normalized_key in TICKET_AGENT_OTEL_PROTECTED_ATTRIBUTE_KEYS:
            continue
        if normalized_key in TICKET_AGENT_OTEL_SENSITIVE_STATE_KEYS:
            continue
        if normalized_key in attributes:
            continue
        safe_value = _safe_otel_attribute_value(value)
        if safe_value is None:
            continue
        attributes[normalized_key] = safe_value


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


def _is_all_zero_hex(value: str) -> bool:
    return set(value) == {"0"}
