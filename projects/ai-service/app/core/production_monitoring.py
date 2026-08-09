from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MetricType = Literal["counter", "gauge", "histogram"]
MetricDomain = Literal[
    "http",
    "llm",
    "rag",
    "tool",
    "java",
    "resilience",
    "cost",
    "safety",
]
AlertSeverity = Literal["critical", "warning", "info"]
AlertComparator = Literal[">", ">=", "<", "<=", "=="]
AlertWindow = Literal["5m", "15m", "30m", "1h", "24h"]

LOW_CARDINALITY_LABELS = frozenset(
    {
        "route",
        "method",
        "status_code_class",
        "flow",
        "provider",
        "model",
        "model_tier",
        "operation",
        "dependency",
        "tool_name",
        "tool_access_level",
        "vector_store",
        "knowledge_base",
        "error_code",
        "fallback_reason",
        "rate_limit_scope",
        "safety_reason",
        "tenant_tier",
    }
)
HIGH_CARDINALITY_LABELS = frozenset(
    {
        "trace_id",
        "span_id",
        "request_id",
        "session_id",
        "thread_id",
        "user_id",
        "actor_id",
        "order_id",
        "ticket_id",
        "email",
        "phone",
        "query",
        "prompt",
        "message",
    }
)

_METRIC_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.]*$")


class ProductionMetricSpec(BaseModel):
    name: str = Field(min_length=1)
    metric_type: MetricType
    domain: MetricDomain
    unit: str = Field(min_length=1)
    description: str = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_metric_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("unit", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def normalize_labels(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="labels")

    @model_validator(mode="after")
    def validate_metric_spec(self) -> "ProductionMetricSpec":
        if not _METRIC_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("metric name must use lowercase dot-separated words")
        forbidden = [label for label in self.labels if label in HIGH_CARDINALITY_LABELS]
        if forbidden:
            raise ValueError(f"metric labels contain high-cardinality values: {', '.join(forbidden)}")
        unknown = [label for label in self.labels if label not in LOW_CARDINALITY_LABELS]
        if unknown:
            raise ValueError(f"metric labels are not in low-cardinality allowlist: {', '.join(unknown)}")
        return self


class AlertRuleSpec(BaseModel):
    name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    severity: AlertSeverity
    comparator: AlertComparator
    threshold: float
    window: AlertWindow
    description: str = Field(min_length=1)
    runbook_hint: str = Field(min_length=1)
    for_duration: str = Field(default="0m", min_length=1)

    @field_validator("name", "metric_name", "description", "runbook_hint", "for_duration", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_alert_rule(self) -> "AlertRuleSpec":
        if not _METRIC_NAME_PATTERN.fullmatch(self.metric_name):
            raise ValueError("alert metric_name must use lowercase dot-separated words")
        if self.window == "5m" and self.severity == "info":
            raise ValueError("5m info alerts are usually too noisy")
        return self


class MonitoringCatalog(BaseModel):
    metrics: list[ProductionMetricSpec] = Field(default_factory=list)
    alert_rules: list[AlertRuleSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_catalog(self) -> "MonitoringCatalog":
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("monitoring catalog metric names must be unique")
        alert_names = [rule.name for rule in self.alert_rules]
        if len(alert_names) != len(set(alert_names)):
            raise ValueError("monitoring catalog alert rule names must be unique")
        unknown_metrics = [
            rule.metric_name
            for rule in self.alert_rules
            if rule.metric_name not in set(metric_names)
        ]
        if unknown_metrics:
            raise ValueError(
                "alert rules reference unknown metrics: "
                + ", ".join(sorted(set(unknown_metrics)))
            )
        return self


class MetricCatalogSummary(BaseModel):
    metric_count: int = Field(ge=0)
    alert_rule_count: int = Field(ge=0)
    domain_counts: dict[str, int] = Field(default_factory=dict)
    type_counts: dict[str, int] = Field(default_factory=dict)
    alert_severity_counts: dict[str, int] = Field(default_factory=dict)
    alert_metric_names: list[str] = Field(default_factory=list)


def build_default_monitoring_catalog() -> MonitoringCatalog:
    metrics = [
        ProductionMetricSpec(
            name="http.server.requests",
            metric_type="counter",
            domain="http",
            unit="requests",
            labels=["route", "method", "status_code_class"],
            description="Total HTTP requests grouped by route, method, and status code class.",
        ),
        ProductionMetricSpec(
            name="http.server.duration",
            metric_type="histogram",
            domain="http",
            unit="ms",
            labels=["route", "method", "status_code_class"],
            description="HTTP request latency distribution for p50, p95, and p99 monitoring.",
        ),
        ProductionMetricSpec(
            name="llm.calls",
            metric_type="counter",
            domain="llm",
            unit="calls",
            labels=["provider", "model", "model_tier", "operation"],
            description="Total LLM calls grouped by provider, model, tier, and operation.",
        ),
        ProductionMetricSpec(
            name="llm.failures",
            metric_type="counter",
            domain="llm",
            unit="failures",
            labels=["provider", "model", "operation", "error_code"],
            description="LLM call failures grouped by stable error code.",
        ),
        ProductionMetricSpec(
            name="llm.fallbacks",
            metric_type="counter",
            domain="resilience",
            unit="fallbacks",
            labels=["provider", "model", "operation", "fallback_reason"],
            description="Fallback decisions and attempts for model failure or degradation.",
        ),
        ProductionMetricSpec(
            name="llm.tokens",
            metric_type="counter",
            domain="cost",
            unit="tokens",
            labels=["provider", "model", "operation"],
            description="Prompt, completion, and total token usage for cost monitoring.",
        ),
        ProductionMetricSpec(
            name="llm.estimated_cost",
            metric_type="counter",
            domain="cost",
            unit="usd",
            labels=["provider", "model", "operation"],
            description="Estimated LLM cost based on configured model pricing.",
        ),
        ProductionMetricSpec(
            name="rag.retrieval.duration",
            metric_type="histogram",
            domain="rag",
            unit="ms",
            labels=["vector_store", "knowledge_base"],
            description="RAG retrieval latency distribution.",
        ),
        ProductionMetricSpec(
            name="rag.retrieval.empty_results",
            metric_type="counter",
            domain="rag",
            unit="cases",
            labels=["vector_store", "knowledge_base"],
            description="RAG retrieval requests that returned no usable context.",
        ),
        ProductionMetricSpec(
            name="rag.citation.failures",
            metric_type="counter",
            domain="rag",
            unit="failures",
            labels=["knowledge_base", "error_code"],
            description="Citation verification failures for RAG answers.",
        ),
        ProductionMetricSpec(
            name="tool.calls",
            metric_type="counter",
            domain="tool",
            unit="calls",
            labels=["tool_name", "tool_access_level"],
            description="Tool calls requested or executed by AI workflows.",
        ),
        ProductionMetricSpec(
            name="tool.failures",
            metric_type="counter",
            domain="tool",
            unit="failures",
            labels=["tool_name", "error_code"],
            description="Tool execution or validation failures grouped by tool and error code.",
        ),
        ProductionMetricSpec(
            name="java.client.duration",
            metric_type="histogram",
            domain="java",
            unit="ms",
            labels=["operation", "error_code"],
            description="Latency distribution for Python-to-Java internal API calls.",
        ),
        ProductionMetricSpec(
            name="rate_limit.rejections",
            metric_type="counter",
            domain="resilience",
            unit="rejections",
            labels=["rate_limit_scope", "route"],
            description="Requests rejected by client, route, AI, or tool rate limits.",
        ),
        ProductionMetricSpec(
            name="safety.blocks",
            metric_type="counter",
            domain="safety",
            unit="blocks",
            labels=["safety_reason", "route"],
            description="Requests blocked by AI safety or privacy boundaries.",
        ),
    ]
    alerts = [
        AlertRuleSpec(
            name="High HTTP 5xx rate",
            metric_name="http.server.requests",
            severity="critical",
            comparator=">=",
            threshold=0.05,
            window="15m",
            for_duration="5m",
            description="More than 5% HTTP requests are 5xx errors in a sustained window.",
            runbook_hint="Check recent deploys, dependency health, exception logs, and trace samples.",
        ),
        AlertRuleSpec(
            name="High p95 request latency",
            metric_name="http.server.duration",
            severity="warning",
            comparator=">=",
            threshold=5000,
            window="15m",
            for_duration="10m",
            description="HTTP p95 latency is above 5 seconds for a sustained window.",
            runbook_hint="Break down latency by LLM, RAG, tool, Java client, and serialization stages.",
        ),
        AlertRuleSpec(
            name="High LLM failure rate",
            metric_name="llm.failures",
            severity="critical",
            comparator=">=",
            threshold=0.1,
            window="15m",
            for_duration="5m",
            description="LLM failures exceed 10% and may affect core AI functionality.",
            runbook_hint="Check provider status, API key validity, timeout, rate limit, and fallback behavior.",
        ),
        AlertRuleSpec(
            name="Cost burn rate high",
            metric_name="llm.estimated_cost",
            severity="warning",
            comparator=">=",
            threshold=10.0,
            window="1h",
            for_duration="0m",
            description="Estimated LLM cost exceeds the configured hourly learning threshold.",
            runbook_hint="Check high-volume routes, token usage, fallback, long prompts, and repeated retries.",
        ),
        AlertRuleSpec(
            name="RAG no-context spike",
            metric_name="rag.retrieval.empty_results",
            severity="warning",
            comparator=">=",
            threshold=0.2,
            window="30m",
            for_duration="10m",
            description="RAG no-context cases increased above expected level.",
            runbook_hint="Check vector store health, collection data, filters, score threshold, and recent ingestion.",
        ),
        AlertRuleSpec(
            name="Safety block spike",
            metric_name="safety.blocks",
            severity="warning",
            comparator=">=",
            threshold=20,
            window="15m",
            for_duration="5m",
            description="Safety blocks increased sharply and may indicate attack traffic or overly strict rules.",
            runbook_hint="Check safety_reason distribution and sampled sanitized request metadata.",
        ),
    ]
    return MonitoringCatalog(metrics=metrics, alert_rules=alerts)


def build_metric_catalog_summary(
    catalog: MonitoringCatalog,
) -> MetricCatalogSummary:
    domain_counts = Counter(metric.domain for metric in catalog.metrics)
    type_counts = Counter(metric.metric_type for metric in catalog.metrics)
    severity_counts = Counter(rule.severity for rule in catalog.alert_rules)
    return MetricCatalogSummary(
        metric_count=len(catalog.metrics),
        alert_rule_count=len(catalog.alert_rules),
        domain_counts=dict(sorted(domain_counts.items())),
        type_counts=dict(sorted(type_counts.items())),
        alert_severity_counts=dict(sorted(severity_counts.items())),
        alert_metric_names=sorted({rule.metric_name for rule in catalog.alert_rules}),
    )


def format_monitoring_catalog(catalog: MonitoringCatalog) -> list[str]:
    summary = build_metric_catalog_summary(catalog)
    lines = [
        "Production monitoring catalog",
        f"metrics: {summary.metric_count}",
        f"alert_rules: {summary.alert_rule_count}",
        f"domains: {_format_counts(summary.domain_counts)}",
        f"metric_types: {_format_counts(summary.type_counts)}",
        f"alert_severities: {_format_counts(summary.alert_severity_counts)}",
        "",
        "Metrics:",
    ]
    lines.extend(
        (
            f"- {metric.name} type={metric.metric_type} domain={metric.domain} "
            f"unit={metric.unit} labels={','.join(metric.labels) or '-'}"
        )
        for metric in catalog.metrics
    )
    lines.extend(["", "Alert rules:"])
    lines.extend(
        (
            f"- {rule.name}: {rule.metric_name} {rule.comparator} "
            f"{rule.threshold} window={rule.window} severity={rule.severity}"
        )
        for rule in catalog.alert_rules
    )
    return lines


def find_high_cardinality_metric_labels(
    catalog: MonitoringCatalog,
) -> dict[str, list[str]]:
    problems: dict[str, list[str]] = {}
    for metric in catalog.metrics:
        labels = [label for label in metric.labels if label in HIGH_CARDINALITY_LABELS]
        if labels:
            problems[metric.name] = labels
    return problems


def evaluate_alert_condition(
    *,
    value: float,
    comparator: AlertComparator,
    threshold: float,
) -> bool:
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == "==":
        return value == threshold
    raise ValueError(f"Unsupported comparator: {comparator}")


def evaluate_alert_rule(
    rule: AlertRuleSpec,
    *,
    current_value: float,
) -> bool:
    return evaluate_alert_condition(
        value=current_value,
        comparator=rule.comparator,
        threshold=rule.threshold,
    )


def _normalize_string_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of strings")
    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-blank strings")
        normalized = item.strip()
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    return normalized_values


def _format_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())
