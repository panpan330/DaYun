from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agents.otel_tracing import (
    OtelAttributeValue,
    build_ticket_agent_otel_resource_attributes,
    build_ticket_agent_otel_span_plan,
)
from app.core.trace import get_trace_id


TicketAgentLogSeverity = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
TicketAgentLogFieldPlacement = Literal[
    "top_level",
    "resource",
    "attribute",
    "forbidden",
]
TicketAgentLogFieldCategory = Literal[
    "time",
    "severity",
    "event_identity",
    "message",
    "resource",
    "correlation",
    "business_context",
    "performance",
    "error",
    "safety",
]
LogFieldValue = str | int | float | bool

TICKET_AGENT_LOG_EVENT_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
)
MAX_LOG_FIELD_STRING_LENGTH = 160

TICKET_AGENT_LOG_SEVERITY_NUMBERS: dict[TicketAgentLogSeverity, int] = {
    "DEBUG": 5,
    "INFO": 9,
    "WARNING": 13,
    "ERROR": 17,
    "CRITICAL": 21,
}

TICKET_AGENT_LOG_FORBIDDEN_FIELD_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "created_ticket",
        "final_answer",
        "messages",
        "normalized_message",
        "order_query_result",
        "password",
        "pending_ticket_confirmation",
        "prompt",
        "rag_answer",
        "rag_citations",
        "rag_query",
        "rag_suggestions",
        "raw_completion",
        "raw_response",
        "ticket_creation_args",
        "ticket_fields",
        "user_message",
    }
)
TICKET_AGENT_LOG_FORBIDDEN_KEY_FRAGMENTS = frozenset(
    {
        "api_key",
        "access_token",
        "refresh_token",
        "secret",
    }
)

TICKET_AGENT_SAFE_STATE_TO_LOG_ATTRIBUTES = (
    ("intent", "agent.intent"),
    ("ticket_need_source", "ticket.need_source"),
    ("order_query_status", "order.query_status"),
    ("order_query_error_code", "order.query_error_code"),
    ("rag_answer_status", "rag.answer_status"),
    ("ticket_field_extraction_source", "ticket.field_extraction_source"),
    ("ticket_fields_complete", "ticket.fields_complete"),
    ("ticket_confirmation_required", "ticket.confirmation_required"),
    ("ticket_confirmation_approved", "ticket.confirmation_approved"),
    ("ticket_tool_name", "ticket.tool_name"),
    ("ticket_tool_access_level", "ticket.tool_access_level"),
    ("ticket_tool_requires_confirmation", "ticket.tool_requires_confirmation"),
    ("ticket_write_safety_status", "ticket.write_safety_status"),
    ("ticket_creation_status", "ticket.creation_status"),
    ("fallback_used", "agent.fallback_used"),
)


@dataclass(frozen=True)
class TicketAgentLogFieldSpec:
    name: str
    placement: TicketAgentLogFieldPlacement
    category: TicketAgentLogFieldCategory
    required: bool
    description: str
    example: LogFieldValue | None = None


@dataclass(frozen=True)
class TicketAgentProductionLogRecord:
    event_name: str
    severity_text: TicketAgentLogSeverity
    severity_number: int
    body: str
    trace_id: str
    span_id: str
    trace_flags: str
    resource: dict[str, LogFieldValue] = field(default_factory=dict)
    attributes: dict[str, LogFieldValue] = field(default_factory=dict)
    timestamp: str | None = None

    def to_otel_log_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "severity_text": self.severity_text,
            "severity_number": self.severity_number,
            "event_name": self.event_name,
            "body": self.body,
            "resource": dict(self.resource),
            "attributes": dict(self.attributes),
        }
        if self.timestamp is not None:
            record["timestamp"] = self.timestamp
        return record

    def flat_fields(self) -> dict[str, LogFieldValue]:
        fields: dict[str, LogFieldValue] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "trace_flags": self.trace_flags,
            "severity_text": self.severity_text,
            "severity_number": self.severity_number,
            "body": self.body,
            "event_name": self.event_name,
        }
        if self.timestamp is not None:
            fields["timestamp"] = self.timestamp
        fields.update(self.resource)
        fields.update(self.attributes)
        return fields


def build_ticket_agent_log_field_specs() -> list[TicketAgentLogFieldSpec]:
    return [
        TicketAgentLogFieldSpec(
            name="timestamp",
            placement="top_level",
            category="time",
            required=True,
            description="日志事件发生时间，真实生产环境通常由日志框架或采集器补齐。",
            example="2026-07-26T10:20:30.123Z",
        ),
        TicketAgentLogFieldSpec(
            name="severity_text",
            placement="top_level",
            category="severity",
            required=True,
            description="日志级别的文本形式，例如 INFO、WARNING、ERROR。",
            example="INFO",
        ),
        TicketAgentLogFieldSpec(
            name="severity_number",
            placement="top_level",
            category="severity",
            required=True,
            description="日志级别的数值形式，用于稳定比较严重程度。",
            example=9,
        ),
        TicketAgentLogFieldSpec(
            name="event_name",
            placement="top_level",
            category="event_identity",
            required=True,
            description="稳定的事件名称，不能放订单号、用户 ID 这类动态值。",
            example="ticket_agent.workflow.succeeded",
        ),
        TicketAgentLogFieldSpec(
            name="body",
            placement="top_level",
            category="message",
            required=True,
            description="给人看的简短说明，不承载大段业务 payload。",
            example="Ticket agent workflow succeeded.",
        ),
        TicketAgentLogFieldSpec(
            name="service.name",
            placement="resource",
            category="resource",
            required=True,
            description="产生日志的服务名。",
            example="ai-service",
        ),
        TicketAgentLogFieldSpec(
            name="deployment.environment.name",
            placement="resource",
            category="resource",
            required=True,
            description="运行环境，例如 local、dev、test、prod。",
            example="local",
        ),
        TicketAgentLogFieldSpec(
            name="trace_id",
            placement="top_level",
            category="correlation",
            required=True,
            description="OpenTelemetry trace id，用于从日志跳到 trace。",
            example="8b0e715c76c8423e9dc95b6c8db8409a",
        ),
        TicketAgentLogFieldSpec(
            name="span_id",
            placement="top_level",
            category="correlation",
            required=True,
            description="当前日志所属 span 的 id。",
            example="5fb397be34d26b51",
        ),
        TicketAgentLogFieldSpec(
            name="app_trace_id",
            placement="attribute",
            category="correlation",
            required=False,
            description="项目早期 X-Trace-Id 或业务请求追踪 id。",
            example="8b0e715c76c8423e9dc95b6c8db8409a",
        ),
        TicketAgentLogFieldSpec(
            name="thread_id",
            placement="attribute",
            category="correlation",
            required=False,
            description="LangGraph 会话或 checkpoint 线程 id，用于跨多次请求串联业务会话。",
            example="ticket-thread-001",
        ),
        TicketAgentLogFieldSpec(
            name="operation",
            placement="attribute",
            category="business_context",
            required=True,
            description="当前执行的稳定操作名。",
            example="invoke_thread",
        ),
        TicketAgentLogFieldSpec(
            name="status",
            placement="attribute",
            category="business_context",
            required=True,
            description="本次事件的结果状态，例如 started、succeeded、failed。",
            example="succeeded",
        ),
        TicketAgentLogFieldSpec(
            name="elapsed_ms",
            placement="attribute",
            category="performance",
            required=False,
            description="本次操作耗时，适合日志排查；聚合趋势应交给 metrics。",
            example=42.13,
        ),
        TicketAgentLogFieldSpec(
            name="error_code",
            placement="attribute",
            category="error",
            required=False,
            description="稳定错误码，错误日志必须优先记录它。",
            example="ORDER_QUERY_TIMEOUT",
        ),
        TicketAgentLogFieldSpec(
            name="user_message",
            placement="forbidden",
            category="safety",
            required=False,
            description="用户原文可能包含手机号、地址、账号等敏感内容，不能进生产日志。",
        ),
        TicketAgentLogFieldSpec(
            name="final_answer",
            placement="forbidden",
            category="safety",
            required=False,
            description="模型完整回答可能包含业务隐私或用户输入复述，不能直接进生产日志。",
        ),
        TicketAgentLogFieldSpec(
            name="order_query_result",
            placement="forbidden",
            category="safety",
            required=False,
            description="订单查询结果可能包含收货人、电话、地址等敏感字段，不能进生产日志。",
        ),
    ]


def validate_ticket_agent_event_name(event_name: str) -> str:
    normalized = event_name.strip()
    if not TICKET_AGENT_LOG_EVENT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Ticket agent event_name must be lowercase, dot-qualified, "
            "and must not contain dynamic values."
        )
    return normalized


def normalize_ticket_agent_log_severity(
    severity_text: TicketAgentLogSeverity | str | None,
    *,
    has_error: bool = False,
) -> TicketAgentLogSeverity:
    if severity_text is None:
        return "ERROR" if has_error else "INFO"

    normalized = severity_text.strip().upper()
    if normalized == "WARN":
        normalized = "WARNING"
    if normalized not in TICKET_AGENT_LOG_SEVERITY_NUMBERS:
        raise ValueError(f"Unsupported ticket agent log severity: {severity_text}")
    return normalized  # type: ignore[return-value]


def find_forbidden_ticket_agent_log_fields(fields: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    _collect_forbidden_log_fields(fields, found=found)
    return sorted(dict.fromkeys(found))


def build_ticket_agent_production_log_record(
    state: Mapping[str, Any],
    *,
    event_name: str,
    operation: str,
    body: str,
    severity_text: TicketAgentLogSeverity | str | None = None,
    status: str | None = None,
    thread_id: str | None = None,
    actor_id: str | None = None,
    incoming_traceparent: str | None = None,
    span_id: str | None = None,
    elapsed_ms: float | None = None,
    timestamp: str | None = None,
    service_version: str | None = None,
    environment: str = "local",
    extra_attributes: Mapping[str, object] | None = None,
) -> TicketAgentProductionLogRecord:
    validated_event_name = validate_ticket_agent_event_name(event_name)
    has_error = _has_agent_error(state) or status == "failed"
    normalized_severity = normalize_ticket_agent_log_severity(
        severity_text,
        has_error=has_error,
    )
    span_plan = build_ticket_agent_otel_span_plan(
        state,
        operation=operation,
        thread_id=thread_id,
        actor_id=actor_id,
        incoming_traceparent=incoming_traceparent,
        span_id=span_id,
        elapsed_ms=elapsed_ms,
    )
    attributes = _build_ticket_agent_log_attributes(
        state,
        operation=operation,
        status=status or _infer_log_status(state),
        thread_id=thread_id,
        actor_id=actor_id,
        span_attributes=span_plan.attributes,
        elapsed_ms=elapsed_ms,
    )
    if extra_attributes is not None:
        _merge_extra_log_attributes(attributes, extra_attributes)

    forbidden_fields = find_forbidden_ticket_agent_log_fields(attributes)
    if forbidden_fields:
        raise ValueError(
            "Forbidden ticket agent log fields: " + ", ".join(forbidden_fields)
        )

    resource = _normalize_resource_attributes(
        build_ticket_agent_otel_resource_attributes(
            service_version=service_version,
            environment=environment,
        )
    )
    return TicketAgentProductionLogRecord(
        event_name=validated_event_name,
        severity_text=normalized_severity,
        severity_number=TICKET_AGENT_LOG_SEVERITY_NUMBERS[normalized_severity],
        body=_normalize_log_body(body),
        trace_id=span_plan.trace_context.trace_id,
        span_id=span_plan.trace_context.span_id,
        trace_flags="01" if span_plan.trace_context.sampled else "00",
        resource=resource,
        attributes=attributes,
        timestamp=timestamp,
    )


def _build_ticket_agent_log_attributes(
    state: Mapping[str, Any],
    *,
    operation: str,
    status: str,
    thread_id: str | None,
    actor_id: str | None,
    span_attributes: Mapping[str, OtelAttributeValue],
    elapsed_ms: float | None,
) -> dict[str, LogFieldValue]:
    attributes: dict[str, LogFieldValue] = {
        "operation": _normalize_required_string(operation, fallback="unknown"),
        "status": _normalize_required_string(status, fallback="unknown"),
        "app_trace_id": _normalize_required_string(
            str(span_attributes.get("app.trace_id") or get_trace_id()),
            fallback="-",
        ),
    }

    _add_optional_log_attribute(attributes, "thread_id", thread_id)
    _add_optional_log_attribute(attributes, "actor_id", actor_id)

    for state_key, attribute_key in TICKET_AGENT_SAFE_STATE_TO_LOG_ATTRIBUTES:
        _add_optional_log_attribute(attributes, attribute_key, state.get(state_key))

    _add_optional_log_attribute(
        attributes,
        "agent.node_last",
        span_attributes.get("agent.node.last"),
    )
    _add_optional_log_attribute(
        attributes,
        "agent.node_count",
        span_attributes.get("agent.node.count"),
    )
    _add_optional_log_attribute(
        attributes,
        "error_code",
        _first_state_value(
            state,
            "agent_error_code",
            "order_query_error_code",
            "ticket_creation_error_code",
        ),
    )
    _add_optional_log_attribute(attributes, "error_node", state.get("agent_error_node"))

    if elapsed_ms is not None and math.isfinite(elapsed_ms):
        attributes["elapsed_ms"] = round(elapsed_ms, 2)

    return attributes


def _merge_extra_log_attributes(
    attributes: dict[str, LogFieldValue],
    extra_attributes: Mapping[str, object],
) -> None:
    forbidden_fields = find_forbidden_ticket_agent_log_fields(extra_attributes)
    if forbidden_fields:
        raise ValueError(
            "Forbidden ticket agent log fields: " + ", ".join(forbidden_fields)
        )

    for key, value in extra_attributes.items():
        normalized_value = _normalize_log_field_value(value)
        if normalized_value is not None:
            attributes[_normalize_required_string(key, fallback="extra")] = (
                normalized_value
            )


def _normalize_resource_attributes(
    attributes: Mapping[str, OtelAttributeValue],
) -> dict[str, LogFieldValue]:
    resource: dict[str, LogFieldValue] = {}
    for key, value in attributes.items():
        normalized_value = _normalize_log_field_value(value)
        if normalized_value is not None:
            resource[key] = normalized_value
    return resource


def _collect_forbidden_log_fields(
    value: Any,
    *,
    found: list[str],
    path: str = "",
) -> None:
    if not isinstance(value, Mapping):
        return

    for key, nested_value in value.items():
        key_text = str(key)
        current_path = f"{path}.{key_text}" if path else key_text
        if _is_forbidden_log_field_key(key_text):
            found.append(current_path)
        _collect_forbidden_log_fields(nested_value, found=found, path=current_path)


def _is_forbidden_log_field_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in TICKET_AGENT_LOG_FORBIDDEN_FIELD_KEYS
        or any(
            fragment in normalized
            for fragment in TICKET_AGENT_LOG_FORBIDDEN_KEY_FRAGMENTS
        )
    )


def _add_optional_log_attribute(
    attributes: dict[str, LogFieldValue],
    key: str,
    value: object,
) -> None:
    normalized_value = _normalize_log_field_value(value)
    if normalized_value is not None:
        attributes[key] = normalized_value


def _normalize_log_field_value(value: object) -> LogFieldValue | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 2)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > MAX_LOG_FIELD_STRING_LENGTH:
            return normalized[:MAX_LOG_FIELD_STRING_LENGTH]
        return normalized
    return None


def _normalize_log_body(body: str) -> str:
    normalized = " ".join(body.strip().split())
    if not normalized:
        return "Ticket agent event."
    return normalized[:MAX_LOG_FIELD_STRING_LENGTH]


def _normalize_required_string(value: object, *, fallback: str) -> str:
    normalized = _normalize_log_field_value(value)
    if isinstance(normalized, str):
        return normalized
    return fallback


def _first_state_value(state: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        value = state.get(key)
        if value is not None:
            return value
    return None


def _has_agent_error(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("agent_error_code")
        or state.get("order_query_error_code")
        or state.get("ticket_creation_error_code")
    )


def _infer_log_status(state: Mapping[str, Any]) -> str:
    if _has_agent_error(state):
        return "failed"
    if state.get("ticket_creation_status") == "blocked":
        return "blocked"
    if state.get("fallback_used") is True:
        return "fallback"
    return "succeeded"
