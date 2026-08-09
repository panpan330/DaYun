from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.core.trace import DEFAULT_TRACE_ID, get_trace_id


AiServiceAttributeValue = str | int | float | bool
AiServiceFlow = Literal["chat", "stream_chat", "rag_answer", "tool_chat"]
AiServiceSpanKind = Literal["SERVER", "INTERNAL", "CLIENT"]
AiServiceSpanStatus = Literal["UNSET", "OK", "ERROR"]
AiServiceEventSeverity = Literal["INFO", "WARNING", "ERROR"]
AiServiceMetricKind = Literal["counter", "histogram", "gauge"]

AI_SERVICE_NAME = "ai-service"

AI_SERVICE_PROTECTED_ATTRIBUTE_KEYS = frozenset(
    {
        "app.trace_id",
        "app.flow",
        "service.name",
        "http.route",
        "http.method",
        "span.name",
    }
)
AI_SERVICE_SENSITIVE_ATTRIBUTE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "set_cookie",
        "llm.api_key",
        "embedding.api_key",
        "rerank.api_key",
        "qdrant.api_key",
        "milvus.token",
        "prompt",
        "raw_prompt",
        "system_prompt",
        "messages",
        "history",
        "user_message",
        "final_answer",
        "raw_response",
        "tool_result",
        "order_payload",
        "ticket_description",
        "document_content",
        "chunk_content",
        "documents",
    }
)
AI_SERVICE_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "parent_span_id",
        "app.trace_id",
        "user_id",
        "actor_id",
        "session_id",
        "thread_id",
        "order_id",
        "ticket_id",
        "request_id",
    }
)

_ATTRIBUTE_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ATTRIBUTE_KEY_UNSAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class AiServiceSpanSpec:
    name: str
    kind: AiServiceSpanKind
    attributes: dict[str, AiServiceAttributeValue]
    parent_name: str | None = None
    status: AiServiceSpanStatus = "UNSET"


@dataclass(frozen=True)
class AiServiceEventSpec:
    name: str
    span_name: str
    severity: AiServiceEventSeverity
    attributes: dict[str, AiServiceAttributeValue]


@dataclass(frozen=True)
class AiServiceMetricSpec:
    name: str
    kind: AiServiceMetricKind
    unit: str
    attributes: dict[str, AiServiceAttributeValue]
    description: str


@dataclass(frozen=True)
class AiServiceTracingPlan:
    trace_id: str
    flow: AiServiceFlow
    root_span: AiServiceSpanSpec
    spans: list[AiServiceSpanSpec]
    events: list[AiServiceEventSpec]
    metrics: list[AiServiceMetricSpec]

    def span_names(self) -> list[str]:
        return [span.name for span in self.spans]

    def event_names(self) -> list[str]:
        return [event.name for event in self.events]

    def metric_names(self) -> list[str]:
        return [metric.name for metric in self.metrics]


def build_ai_service_span_attributes(
    *,
    flow: AiServiceFlow,
    operation: str,
    trace_id: str | None = None,
    route: str | None = None,
    method: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    tool_name: str | None = None,
    vector_store: str | None = None,
    collection: str | None = None,
    top_k: int | None = None,
    timeout_seconds: float | None = None,
    extra_attributes: Mapping[str, object] | None = None,
) -> dict[str, AiServiceAttributeValue]:
    attributes: dict[str, AiServiceAttributeValue] = {}
    _add_attribute(attributes, "service.name", AI_SERVICE_NAME)
    _add_attribute(attributes, "app.flow", flow)
    _add_attribute(attributes, "app.operation", operation)
    _add_attribute(attributes, "app.trace_id", _resolve_trace_id(trace_id))
    _add_attribute(attributes, "http.route", route)
    _add_attribute(attributes, "http.method", method)
    _add_attribute(attributes, "llm.model", model)
    _add_attribute(attributes, "llm.provider", provider)
    _add_attribute(attributes, "tool.name", tool_name)
    _add_attribute(attributes, "vector.store", vector_store)
    _add_attribute(attributes, "vector.collection", collection)
    _add_attribute(attributes, "rag.top_k", top_k)
    _add_attribute(attributes, "app.timeout_seconds", timeout_seconds)

    if extra_attributes is not None:
        _merge_extra_attributes(attributes, extra_attributes)

    return attributes


def build_python_ai_service_tracing_plan(
    *,
    flow: AiServiceFlow,
    trace_id: str | None = None,
    route: str | None = None,
    method: str = "POST",
    model: str | None = None,
    provider: str | None = None,
    tool_name: str | None = None,
    vector_store: str | None = None,
    collection: str | None = None,
    top_k: int | None = None,
    include_rerank: bool = True,
) -> AiServiceTracingPlan:
    selected_route = route or _default_route_for_flow(flow)
    selected_trace_id = _resolve_trace_id(trace_id)
    root_span = _span(
        name="http.request",
        kind="SERVER",
        flow=flow,
        operation="http.request",
        trace_id=selected_trace_id,
        route=selected_route,
        method=method,
    )
    spans = [
        root_span,
        _span(
            name="request.validation",
            kind="INTERNAL",
            flow=flow,
            operation="request.validation",
            trace_id=selected_trace_id,
            parent_name=root_span.name,
            route=selected_route,
            method=method,
        ),
    ]

    if flow == "chat":
        spans.extend(
            [
                _span(
                    name="prompt.build",
                    kind="INTERNAL",
                    flow=flow,
                    operation="prompt.build",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                ),
                _span(
                    name="llm.call",
                    kind="CLIENT",
                    flow=flow,
                    operation="llm.call",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
            ]
        )

    if flow == "stream_chat":
        spans.extend(
            [
                _span(
                    name="prompt.build",
                    kind="INTERNAL",
                    flow=flow,
                    operation="prompt.build",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                ),
                _span(
                    name="llm.stream",
                    kind="CLIENT",
                    flow=flow,
                    operation="llm.stream",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
                _span(
                    name="sse.stream",
                    kind="SERVER",
                    flow=flow,
                    operation="sse.stream",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                ),
            ]
        )

    if flow == "rag_answer":
        spans.extend(
            [
                _span(
                    name="rag.query_rewrite",
                    kind="INTERNAL",
                    flow=flow,
                    operation="rag.query_rewrite",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                ),
                _span(
                    name="embedding.call",
                    kind="CLIENT",
                    flow=flow,
                    operation="embedding.call",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
                _span(
                    name="vector.search",
                    kind="CLIENT",
                    flow=flow,
                    operation="vector.search",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    vector_store=vector_store,
                    collection=collection,
                    top_k=top_k,
                ),
            ]
        )
        if include_rerank:
            spans.append(
                _span(
                    name="rerank.call",
                    kind="CLIENT",
                    flow=flow,
                    operation="rerank.call",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                )
            )
        spans.extend(
            [
                _span(
                    name="context.compression",
                    kind="INTERNAL",
                    flow=flow,
                    operation="context.compression",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                ),
                _span(
                    name="llm.final_answer",
                    kind="CLIENT",
                    flow=flow,
                    operation="llm.final_answer",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
            ]
        )

    if flow == "tool_chat":
        spans.extend(
            [
                _span(
                    name="llm.tool_decision",
                    kind="CLIENT",
                    flow=flow,
                    operation="llm.tool_decision",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
                _span(
                    name="tool.validation",
                    kind="INTERNAL",
                    flow=flow,
                    operation="tool.validation",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    tool_name=tool_name,
                ),
                _span(
                    name="tool.execution",
                    kind="INTERNAL",
                    flow=flow,
                    operation="tool.execution",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    tool_name=tool_name,
                ),
                _span(
                    name="java.orders.get",
                    kind="CLIENT",
                    flow=flow,
                    operation="java.orders.get",
                    trace_id=selected_trace_id,
                    parent_name="tool.execution",
                    route=selected_route,
                    method=method,
                    tool_name=tool_name,
                ),
                _span(
                    name="llm.final_answer",
                    kind="CLIENT",
                    flow=flow,
                    operation="llm.final_answer",
                    trace_id=selected_trace_id,
                    parent_name=root_span.name,
                    route=selected_route,
                    method=method,
                    model=model,
                    provider=provider,
                ),
            ]
        )

    return AiServiceTracingPlan(
        trace_id=selected_trace_id,
        flow=flow,
        root_span=root_span,
        spans=spans,
        events=_build_event_specs(flow),
        metrics=_build_metric_specs(flow, route=selected_route, model=model, provider=provider),
    )


def _span(
    *,
    name: str,
    kind: AiServiceSpanKind,
    flow: AiServiceFlow,
    operation: str,
    trace_id: str,
    parent_name: str | None = None,
    route: str | None = None,
    method: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    tool_name: str | None = None,
    vector_store: str | None = None,
    collection: str | None = None,
    top_k: int | None = None,
) -> AiServiceSpanSpec:
    attributes = build_ai_service_span_attributes(
        flow=flow,
        operation=operation,
        trace_id=trace_id,
        route=route,
        method=method,
        model=model,
        provider=provider,
        tool_name=tool_name,
        vector_store=vector_store,
        collection=collection,
        top_k=top_k,
        extra_attributes={"span.name": name},
    )
    return AiServiceSpanSpec(
        name=name,
        kind=kind,
        parent_name=parent_name,
        attributes=attributes,
    )


def _build_event_specs(flow: AiServiceFlow) -> list[AiServiceEventSpec]:
    if flow == "stream_chat":
        return [
            _event("sse_client_disconnected", "sse.stream", "WARNING"),
            _event("stream_error_sent", "sse.stream", "ERROR"),
            _event("timeout", "llm.stream", "ERROR"),
        ]
    if flow == "rag_answer":
        return [
            _event("prompt_injection_detected", "http.request", "WARNING"),
            _event("rag_no_relevant_context", "vector.search", "WARNING"),
            _event("rerank_fallback_used", "rerank.call", "WARNING"),
            _event("citation_verification_failed", "llm.final_answer", "WARNING"),
        ]
    if flow == "tool_chat":
        return [
            _event("tool_requested", "llm.tool_decision", "INFO"),
            _event("tool_validation_failed", "tool.validation", "ERROR"),
            _event("permission_denied", "java.orders.get", "WARNING"),
            _event("timeout", "java.orders.get", "ERROR"),
            _event("fallback_triggered", "llm.final_answer", "WARNING"),
        ]
    return [
        _event("llm_timeout", "llm.call", "ERROR"),
        _event("fallback_triggered", "llm.call", "WARNING"),
    ]


def _event(
    name: str,
    span_name: str,
    severity: AiServiceEventSeverity,
) -> AiServiceEventSpec:
    return AiServiceEventSpec(
        name=name,
        span_name=span_name,
        severity=severity,
        attributes={"event.name": name, "event.severity": severity},
    )


def _build_metric_specs(
    flow: AiServiceFlow,
    *,
    route: str,
    model: str | None,
    provider: str | None,
) -> list[AiServiceMetricSpec]:
    metric_attributes = _build_low_cardinality_metric_attributes(
        {
            "app.flow": flow,
            "http.route": route,
            "llm.model": model,
            "llm.provider": provider,
        }
    )
    metrics = [
        AiServiceMetricSpec(
            name="ai_service.request.count",
            kind="counter",
            unit="1",
            attributes=metric_attributes,
            description="Number of AI service requests.",
        ),
        AiServiceMetricSpec(
            name="ai_service.request.duration",
            kind="histogram",
            unit="ms",
            attributes=metric_attributes,
            description="Distribution of AI service request latency.",
        ),
    ]
    if flow in {"chat", "stream_chat", "rag_answer", "tool_chat"}:
        metrics.append(
            AiServiceMetricSpec(
                name="ai_service.llm.calls",
                kind="counter",
                unit="1",
                attributes=metric_attributes,
                description="Number of LLM calls from the Python AI service.",
            )
        )
    if flow == "rag_answer":
        metrics.extend(
            [
                AiServiceMetricSpec(
                    name="ai_service.rag.retrieval.duration",
                    kind="histogram",
                    unit="ms",
                    attributes=metric_attributes,
                    description="Distribution of RAG retrieval latency.",
                ),
                AiServiceMetricSpec(
                    name="ai_service.rag.empty_result.count",
                    kind="counter",
                    unit="1",
                    attributes=metric_attributes,
                    description="Number of RAG retrievals without relevant context.",
                ),
            ]
        )
    if flow == "tool_chat":
        metrics.extend(
            [
                AiServiceMetricSpec(
                    name="ai_service.tool.calls",
                    kind="counter",
                    unit="1",
                    attributes=metric_attributes,
                    description="Number of backend tool calls requested by the model.",
                ),
                AiServiceMetricSpec(
                    name="ai_service.java.client.duration",
                    kind="histogram",
                    unit="ms",
                    attributes=metric_attributes,
                    description="Distribution of Python to Java client latency.",
                ),
            ]
        )
    return metrics


def _build_low_cardinality_metric_attributes(
    attributes: Mapping[str, object],
) -> dict[str, AiServiceAttributeValue]:
    metric_attributes: dict[str, AiServiceAttributeValue] = {}
    for key, value in attributes.items():
        normalized_key = _normalize_attribute_key(key)
        if normalized_key is None:
            continue
        if normalized_key in AI_SERVICE_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS:
            continue
        if normalized_key in AI_SERVICE_SENSITIVE_ATTRIBUTE_KEYS:
            continue
        safe_value = _safe_attribute_value(value)
        if safe_value is not None:
            metric_attributes[normalized_key] = safe_value
    return metric_attributes


def _resolve_trace_id(trace_id: str | None) -> str:
    if trace_id is not None and trace_id.strip():
        return trace_id.strip()
    current_trace_id = get_trace_id()
    if current_trace_id != DEFAULT_TRACE_ID:
        return current_trace_id
    return DEFAULT_TRACE_ID


def _default_route_for_flow(flow: AiServiceFlow) -> str:
    if flow == "stream_chat":
        return "/stream-chat"
    if flow == "rag_answer":
        return "/rag/query"
    if flow == "tool_chat":
        return "/tool-chat"
    return "/chat"


def _normalize_attribute_key(key: object) -> str | None:
    if key is None:
        return None
    text = str(key).strip().replace(" ", "_")
    if not text:
        return None
    normalized = _ATTRIBUTE_KEY_UNSAFE_PATTERN.sub("_", text).strip("_.-").casefold()
    if not normalized:
        return None
    if not _ATTRIBUTE_KEY_PATTERN.fullmatch(normalized):
        return None
    return normalized


def _safe_attribute_value(value: object) -> AiServiceAttributeValue | None:
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


def _add_attribute(
    attributes: dict[str, AiServiceAttributeValue],
    key: object,
    value: object,
) -> None:
    normalized_key = _normalize_attribute_key(key)
    if normalized_key is None:
        return
    if normalized_key in AI_SERVICE_SENSITIVE_ATTRIBUTE_KEYS:
        return
    safe_value = _safe_attribute_value(value)
    if safe_value is not None:
        attributes[normalized_key] = safe_value


def _merge_extra_attributes(
    attributes: dict[str, AiServiceAttributeValue],
    extra_attributes: Mapping[str, object],
) -> None:
    for key, value in extra_attributes.items():
        normalized_key = _normalize_attribute_key(key)
        if normalized_key is None:
            continue
        if normalized_key in AI_SERVICE_PROTECTED_ATTRIBUTE_KEYS:
            continue
        if normalized_key in AI_SERVICE_SENSITIVE_ATTRIBUTE_KEYS:
            continue
        if normalized_key in attributes:
            continue
        safe_value = _safe_attribute_value(value)
        if safe_value is not None:
            attributes[normalized_key] = safe_value
