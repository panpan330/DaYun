from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agents.otel_tracing import (
    OtelAttributeValue,
    TicketAgentOtelSpanPlan,
    build_ticket_agent_otel_span_plan,
)
from app.core.trace import get_trace_id


ObservabilitySignalType = Literal["trace", "span", "log", "metric"]
MetricInstrumentKind = Literal["counter", "histogram", "gauge", "up_down_counter"]
LogSeverity = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
TicketAgentTroubleshootingSymptom = Literal[
    "one_user_failed",
    "latency_regression",
    "error_rate_regression",
    "agent_decision_debug",
]

TICKET_AGENT_INVOCATION_METRIC_NAME = "ticket_agent.invocations"
TICKET_AGENT_ERROR_METRIC_NAME = "ticket_agent.errors"
TICKET_AGENT_DURATION_METRIC_NAME = "ticket_agent.duration"
TICKET_AGENT_NODE_COUNT_METRIC_NAME = "ticket_agent.node.count"

HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "otel_trace_id",
        "thread_id",
        "session_id",
        "actor_id",
        "app.trace_id",
        "app.thread_id",
        "app.session_id",
        "app.actor_id",
    }
)


@dataclass(frozen=True)
class TicketAgentSignalCorrelation:
    app_trace_id: str
    otel_trace_id: str
    span_id: str
    thread_id: str | None = None
    actor_id: str | None = None

    def log_fields(self) -> dict[str, str]:
        fields = {
            "trace_id": self.app_trace_id,
            "otel_trace_id": self.otel_trace_id,
            "span_id": self.span_id,
        }
        if self.thread_id is not None:
            fields["thread_id"] = self.thread_id
        if self.actor_id is not None:
            fields["actor_id"] = self.actor_id
        return fields


@dataclass(frozen=True)
class TicketAgentTraceSignal:
    trace_id: str
    app_trace_id: str
    root_operation: str
    thread_id: str | None
    span_count: int
    status: str
    purpose: str = "show_request_path"


@dataclass(frozen=True)
class TicketAgentSpanSignal:
    name: str
    span_id: str
    parent_span_id: str | None
    status: str
    kind: str
    attributes: dict[str, OtelAttributeValue]
    purpose: str = "show_operation_timing_and_attributes"


@dataclass(frozen=True)
class TicketAgentLogSignal:
    event_name: str
    severity: LogSeverity
    message_template: str
    fields: dict[str, OtelAttributeValue | str]
    purpose: str = "explain_point_in_time_event"


@dataclass(frozen=True)
class TicketAgentMetricSignal:
    name: str
    kind: MetricInstrumentKind
    value: int | float
    unit: str
    attributes: dict[str, OtelAttributeValue]
    description: str
    purpose: str = "show_aggregate_system_behavior"


@dataclass(frozen=True)
class TicketAgentInvestigationStep:
    order: int
    signal_type: ObservabilitySignalType
    question: str
    fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TicketAgentObservabilitySignals:
    correlation: TicketAgentSignalCorrelation
    trace: TicketAgentTraceSignal
    span: TicketAgentSpanSignal
    logs: list[TicketAgentLogSignal]
    metrics: list[TicketAgentMetricSignal]

    def log_event_names(self) -> list[str]:
        return [log.event_name for log in self.logs]

    def metric_names(self) -> list[str]:
        return [metric.name for metric in self.metrics]


def build_ticket_agent_observability_signals(
    state: Mapping[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    actor_id: str | None = None,
    incoming_traceparent: str | None = None,
    span_id: str | None = None,
    elapsed_ms: float | None = None,
) -> TicketAgentObservabilitySignals:
    span_plan = build_ticket_agent_otel_span_plan(
        state,
        operation=operation,
        thread_id=thread_id,
        actor_id=actor_id,
        incoming_traceparent=incoming_traceparent,
        span_id=span_id,
        elapsed_ms=elapsed_ms,
    )
    correlation = _build_correlation(
        state,
        span_plan=span_plan,
        thread_id=thread_id,
        actor_id=actor_id,
    )
    trace = _build_trace_signal(
        span_plan=span_plan,
        operation=operation,
        correlation=correlation,
    )
    span = _build_span_signal(span_plan)
    logs = _build_log_signals(
        span_plan=span_plan,
        operation=operation,
        correlation=correlation,
        elapsed_ms=elapsed_ms,
    )
    metrics = _build_metric_signals(
        span_plan=span_plan,
        operation=operation,
        elapsed_ms=elapsed_ms,
    )
    return TicketAgentObservabilitySignals(
        correlation=correlation,
        trace=trace,
        span=span,
        logs=logs,
        metrics=metrics,
    )


def build_ticket_agent_investigation_steps(
    symptom: TicketAgentTroubleshootingSymptom,
) -> list[TicketAgentInvestigationStep]:
    if symptom == "one_user_failed":
        return [
            TicketAgentInvestigationStep(
                order=1,
                signal_type="log",
                question="Find the exact request or user report by trace_id/thread_id.",
                fields=["trace_id", "thread_id", "actor_id", "error_code"],
            ),
            TicketAgentInvestigationStep(
                order=2,
                signal_type="trace",
                question="Open the request path and see which operation failed.",
                fields=["otel_trace_id", "span_id", "status"],
            ),
            TicketAgentInvestigationStep(
                order=3,
                signal_type="span",
                question="Inspect the failing span attributes and status.",
                fields=["agent.error_code", "agent.error_node", "ticket.creation.status"],
            ),
            TicketAgentInvestigationStep(
                order=4,
                signal_type="metric",
                question="Check whether this is isolated or part of a wider failure.",
                fields=["ticket_agent.errors", "ticket_agent.invocations"],
            ),
        ]

    if symptom == "latency_regression":
        return [
            TicketAgentInvestigationStep(
                order=1,
                signal_type="metric",
                question="Confirm latency trend and affected operation from histograms.",
                fields=["ticket_agent.duration", "operation", "intent"],
            ),
            TicketAgentInvestigationStep(
                order=2,
                signal_type="trace",
                question="Sample slow traces and compare the request path.",
                fields=["otel_trace_id", "span_count"],
            ),
            TicketAgentInvestigationStep(
                order=3,
                signal_type="span",
                question="Find the slow child operation or Agent step.",
                fields=["span_id", "app.elapsed_ms", "agent.node.last"],
            ),
            TicketAgentInvestigationStep(
                order=4,
                signal_type="log",
                question="Use logs for exact decisions or upstream error details.",
                fields=["trace_id", "event_name", "error_code"],
            ),
        ]

    if symptom == "error_rate_regression":
        return [
            TicketAgentInvestigationStep(
                order=1,
                signal_type="metric",
                question="Confirm error rate trend and which operation is affected.",
                fields=["ticket_agent.errors", "operation", "intent", "error_code"],
            ),
            TicketAgentInvestigationStep(
                order=2,
                signal_type="trace",
                question="Open representative failing traces.",
                fields=["otel_trace_id", "status"],
            ),
            TicketAgentInvestigationStep(
                order=3,
                signal_type="span",
                question="Identify the failing span and business status.",
                fields=["agent.error_code", "ticket.creation.error_code"],
            ),
            TicketAgentInvestigationStep(
                order=4,
                signal_type="log",
                question="Read exact error logs and fallback decisions.",
                fields=["trace_id", "fallback_used", "error_code"],
            ),
        ]

    return [
        TicketAgentInvestigationStep(
            order=1,
            signal_type="trace",
            question="Follow the Agent route for this single execution.",
            fields=["otel_trace_id", "span_id", "thread_id"],
        ),
        TicketAgentInvestigationStep(
            order=2,
            signal_type="span",
            question="Inspect intent, route, node history summary, and business attributes.",
            fields=["agent.intent", "agent.node.last", "ticket.write_safety.status"],
        ),
        TicketAgentInvestigationStep(
            order=3,
            signal_type="log",
            question="Read point-in-time decision logs for the route explanation.",
            fields=["trace_id", "event_name", "intent", "last_node"],
        ),
        TicketAgentInvestigationStep(
            order=4,
            signal_type="metric",
            question="Use metrics only to compare whether this decision pattern is common.",
            fields=["ticket_agent.invocations", "intent", "status"],
        ),
    ]


def _build_correlation(
    state: Mapping[str, Any],
    *,
    span_plan: TicketAgentOtelSpanPlan,
    thread_id: str | None,
    actor_id: str | None,
) -> TicketAgentSignalCorrelation:
    attributes = span_plan.attributes
    app_trace_id = _string_attribute(attributes.get("app.trace_id"))
    selected_thread_id = _string_attribute(attributes.get("app.thread_id")) or thread_id
    selected_actor_id = _string_attribute(attributes.get("app.actor_id")) or actor_id
    return TicketAgentSignalCorrelation(
        app_trace_id=app_trace_id or str(state.get("agent_trace_id") or get_trace_id()),
        otel_trace_id=span_plan.trace_context.trace_id,
        span_id=span_plan.trace_context.span_id,
        thread_id=selected_thread_id,
        actor_id=selected_actor_id,
    )


def _build_trace_signal(
    *,
    span_plan: TicketAgentOtelSpanPlan,
    operation: str,
    correlation: TicketAgentSignalCorrelation,
) -> TicketAgentTraceSignal:
    return TicketAgentTraceSignal(
        trace_id=correlation.otel_trace_id,
        app_trace_id=correlation.app_trace_id,
        root_operation=operation,
        thread_id=correlation.thread_id,
        span_count=1,
        status=span_plan.status,
    )


def _build_span_signal(
    span_plan: TicketAgentOtelSpanPlan,
) -> TicketAgentSpanSignal:
    return TicketAgentSpanSignal(
        name=span_plan.span_name,
        span_id=span_plan.trace_context.span_id,
        parent_span_id=span_plan.trace_context.parent_span_id,
        status=span_plan.status,
        kind=span_plan.span_kind,
        attributes=dict(span_plan.attributes),
    )


def _build_log_signals(
    *,
    span_plan: TicketAgentOtelSpanPlan,
    operation: str,
    correlation: TicketAgentSignalCorrelation,
    elapsed_ms: float | None,
) -> list[TicketAgentLogSignal]:
    attributes = span_plan.attributes
    started_fields: dict[str, OtelAttributeValue | str] = {
        **correlation.log_fields(),
        "operation": operation,
    }
    _copy_optional_attribute(started_fields, attributes, "agent.intent", "intent")

    finished_fields: dict[str, OtelAttributeValue | str] = {
        **correlation.log_fields(),
        "operation": operation,
        "status": _operation_status(span_plan),
    }
    if elapsed_ms is not None:
        finished_fields["elapsed_ms"] = round(elapsed_ms, 2)
    _copy_optional_attribute(finished_fields, attributes, "agent.intent", "intent")
    _copy_optional_attribute(
        finished_fields,
        attributes,
        "agent.node.last",
        "last_node",
    )
    _copy_optional_attribute(
        finished_fields,
        attributes,
        "agent.fallback_used",
        "fallback_used",
    )

    logs = [
        TicketAgentLogSignal(
            event_name="ticket_agent_started",
            severity="INFO",
            message_template="ticket_agent_started operation={operation}",
            fields=started_fields,
        )
    ]

    if span_plan.status == "ERROR":
        error_fields = {
            **finished_fields,
            "error_code": span_plan.status_description or "UNKNOWN_ERROR",
        }
        logs.append(
            TicketAgentLogSignal(
                event_name="ticket_agent_failed",
                severity="ERROR",
                message_template=(
                    "ticket_agent_failed operation={operation} error_code={error_code}"
                ),
                fields=error_fields,
            )
        )
    else:
        logs.append(
            TicketAgentLogSignal(
                event_name="ticket_agent_finished",
                severity="INFO",
                message_template="ticket_agent_finished operation={operation}",
                fields=finished_fields,
            )
        )

    return logs


def _build_metric_signals(
    *,
    span_plan: TicketAgentOtelSpanPlan,
    operation: str,
    elapsed_ms: float | None,
) -> list[TicketAgentMetricSignal]:
    attributes = span_plan.attributes
    metric_attributes = _build_low_cardinality_metric_attributes(
        attributes,
        operation=operation,
        status=_operation_status(span_plan),
    )
    metrics = [
        TicketAgentMetricSignal(
            name=TICKET_AGENT_INVOCATION_METRIC_NAME,
            kind="counter",
            value=1,
            unit="1",
            attributes=metric_attributes,
            description="Number of ticket agent invocations.",
        )
    ]

    if span_plan.status == "ERROR":
        error_attributes = dict(metric_attributes)
        if span_plan.status_description:
            error_attributes["error_code"] = span_plan.status_description
        metrics.append(
            TicketAgentMetricSignal(
                name=TICKET_AGENT_ERROR_METRIC_NAME,
                kind="counter",
                value=1,
                unit="1",
                attributes=error_attributes,
                description="Number of failed ticket agent invocations.",
            )
        )

    if elapsed_ms is not None:
        metrics.append(
            TicketAgentMetricSignal(
                name=TICKET_AGENT_DURATION_METRIC_NAME,
                kind="histogram",
                value=round(elapsed_ms, 2),
                unit="ms",
                attributes=metric_attributes,
                description="Distribution of ticket agent invocation duration.",
            )
        )

    node_count = attributes.get("agent.node.count")
    if isinstance(node_count, int):
        metrics.append(
            TicketAgentMetricSignal(
                name=TICKET_AGENT_NODE_COUNT_METRIC_NAME,
                kind="histogram",
                value=node_count,
                unit="1",
                attributes=metric_attributes,
                description="Distribution of executed Agent node counts.",
            )
        )

    return metrics


def _build_low_cardinality_metric_attributes(
    attributes: Mapping[str, OtelAttributeValue],
    *,
    operation: str,
    status: str,
) -> dict[str, OtelAttributeValue]:
    metric_attributes: dict[str, OtelAttributeValue] = {
        "operation": operation,
        "status": status,
    }
    for source_key, target_key in (
        ("agent.intent", "intent"),
        ("ticket.creation.status", "ticket_creation_status"),
        ("ticket.write_safety.status", "ticket_write_safety_status"),
        ("rag.answer.status", "rag_answer_status"),
        ("order.query.status", "order_query_status"),
    ):
        value = attributes.get(source_key)
        if value is not None:
            metric_attributes[target_key] = value

    return {
        key: value
        for key, value in metric_attributes.items()
        if key not in HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS
    }


def _operation_status(span_plan: TicketAgentOtelSpanPlan) -> str:
    return "error" if span_plan.status == "ERROR" else "ok"


def _copy_optional_attribute(
    target: dict[str, OtelAttributeValue | str],
    source: Mapping[str, OtelAttributeValue],
    source_key: str,
    target_key: str,
) -> None:
    value = source.get(source_key)
    if value is not None:
        target[target_key] = value


def _string_attribute(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
