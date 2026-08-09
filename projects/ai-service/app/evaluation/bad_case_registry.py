from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.bad_case_analysis import BadCaseAnalysisItem


BadCaseSource = Literal["eval", "production", "manual"]
BadCaseTaskType = Literal["agent", "rag", "tool_calling", "safety", "unknown"]
BadCaseSeverity = Literal["critical", "high", "medium", "low"]
BadCaseStatus = Literal[
    "open",
    "triaged",
    "fixed",
    "regression_added",
    "closed",
]
BadCaseFailureLayer = Literal[
    "intent",
    "field_extraction",
    "routing",
    "rag_retrieval",
    "rag_citation",
    "agent_decision",
    "tool_calling",
    "permission",
    "security",
    "model_output",
    "data",
    "unknown",
]
ProductionRegressionAssertion = Literal[
    "intent",
    "citation_present",
    "ticket_confirmation_required",
    "tool_called",
    "must_ask_for",
    "must_not_reveal",
]


class ProductionRegressionSpec(BaseModel):
    """A supervisor-approved, deterministic check for one production bad case."""

    message: str = Field(min_length=1, max_length=4000)
    assertion: ProductionRegressionAssertion
    expected_intent: Literal[
        "policy_question",
        "order_query",
        "ticket_request",
        "refund_request",
        "smalltalk",
        "unsupported",
        "unclear",
    ] | None = None
    expected_tool: str | None = Field(
        default=None, description="tool_called 断言：期望被调用的工具名"
    )
    must_ask_fields: list[str] = Field(
        default_factory=list, description="must_ask_for 断言：必须追问的字段"
    )
    must_not_reveal_terms: list[str] = Field(
        default_factory=list, description="must_not_reveal 断言：不得泄露的 term"
    )

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_assertion_fields(self) -> "ProductionRegressionSpec":
        if self.assertion == "intent" and self.expected_intent is None:
            raise ValueError("expected_intent is required for an intent regression assertion")
        if self.assertion != "intent" and self.expected_intent is not None:
            raise ValueError("expected_intent is only supported by an intent regression assertion")
        if self.assertion == "tool_called" and not self.expected_tool:
            raise ValueError("expected_tool is required for a tool_called regression assertion")
        if self.assertion != "tool_called" and self.expected_tool is not None:
            raise ValueError("expected_tool is only supported by a tool_called regression assertion")
        if self.assertion == "must_ask_for" and not self.must_ask_fields:
            raise ValueError("must_ask_fields is required for a must_ask_for regression assertion")
        if self.assertion != "must_ask_for" and self.must_ask_fields:
            raise ValueError("must_ask_fields is only supported by a must_ask_for regression assertion")
        if self.assertion == "must_not_reveal" and not self.must_not_reveal_terms:
            raise ValueError(
                "must_not_reveal_terms is required for a must_not_reveal regression assertion"
            )
        if self.assertion != "must_not_reveal" and self.must_not_reveal_terms:
            raise ValueError(
                "must_not_reveal_terms is only supported by a must_not_reveal regression assertion"
            )
        return self


class BadCaseRecord(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: BadCaseSource
    task_type: BadCaseTaskType = "unknown"
    severity: BadCaseSeverity
    status: BadCaseStatus = "open"
    discovered_run_id: str | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    source_case_id: str | None = None
    failure_layer: BadCaseFailureLayer = "unknown"
    failure_category: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    actual_behavior: str = Field(min_length=1)
    root_cause: str = ""
    recommended_action: str = Field(min_length=1)
    regression_action: str = Field(min_length=1)
    regression_dataset_name: str | None = None
    regression_case_id: str | None = None
    production_regression: ProductionRegressionSpec | None = None
    evidence_summary: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator(
        "id",
        "title",
        "discovered_run_id",
        "dataset_name",
        "dataset_version",
        "source_case_id",
        "failure_category",
        "expected_behavior",
        "actual_behavior",
        "root_cause",
        "recommended_action",
        "regression_action",
        "regression_dataset_name",
        "regression_case_id",
        "evidence_summary",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="tags")


class BadCaseRegistry(BaseModel):
    schema_version: str = Field(min_length=1)
    records: list[BadCaseRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_record_ids(self) -> "BadCaseRegistry":
        ids = [record.id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("bad case registry record ids must be unique")
        return self


class BadCaseRegistrySummary(BaseModel):
    record_count: int = Field(ge=0)
    open_count: int = Field(ge=0)
    regression_added_count: int = Field(ge=0)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    layer_counts: dict[str, int] = Field(default_factory=dict)


class RegressionCaseDraft(BaseModel):
    source_bad_case_id: str = Field(min_length=1)
    target_dataset_name: str = Field(min_length=1)
    suggested_case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    input_summary: str = Field(min_length=1)
    expected_behavior: str = Field(min_length=1)
    assertions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def build_bad_case_registry_summary(
    registry: BadCaseRegistry,
) -> BadCaseRegistrySummary:
    severity_counts = Counter(record.severity for record in registry.records)
    status_counts = Counter(record.status for record in registry.records)
    layer_counts = Counter(record.failure_layer for record in registry.records)

    return BadCaseRegistrySummary(
        record_count=len(registry.records),
        open_count=status_counts["open"],
        regression_added_count=status_counts["regression_added"],
        severity_counts=dict(sorted(severity_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
        layer_counts=dict(sorted(layer_counts.items())),
    )


def format_bad_case_registry_summary(summary: BadCaseRegistrySummary) -> list[str]:
    return [
        "Bad case registry summary",
        f"records: {summary.record_count}",
        f"open: {summary.open_count}",
        f"regression_added: {summary.regression_added_count}",
        f"severity: {_format_counts(summary.severity_counts)}",
        f"status: {_format_counts(summary.status_counts)}",
        f"layers: {_format_counts(summary.layer_counts)}",
    ]


def build_bad_case_record_from_analysis_item(
    item: BadCaseAnalysisItem,
    *,
    discovered_run_id: str,
    dataset_name: str,
    dataset_version: str,
) -> BadCaseRecord:
    failure_layer = _map_analysis_category_to_failure_layer(item.category)
    source_case_id = item.case_id if item.case_id != "unknown_case" else None
    bad_case_id = build_bad_case_id(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        source_case_id=source_case_id or "unknown",
        failure_layer=failure_layer,
    )

    return BadCaseRecord(
        id=bad_case_id,
        title=f"{item.suite_name} bad case: {item.case_id}",
        source="eval",
        task_type=_task_type_from_suite(item.suite_name),
        severity=_severity_from_priority(item.priority),
        status="open",
        discovered_run_id=discovered_run_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        source_case_id=source_case_id,
        failure_layer=failure_layer,
        failure_category=item.category,
        expected_behavior=_extract_expected_behavior(item.evidence_lines),
        actual_behavior=_extract_actual_behavior(item.evidence_lines),
        root_cause=item.diagnosis,
        recommended_action=item.recommended_action,
        regression_action=item.regression_action,
        regression_dataset_name=dataset_name,
        evidence_summary=_summarize_evidence(item.evidence_lines),
        tags=_unique_strings(
            [
                "bad_case",
                "regression_candidate",
                item.suite_name,
                item.category,
                failure_layer,
                *(["p0"] if item.priority == "p0" else []),
            ]
        ),
    )


def build_bad_case_id(
    *,
    dataset_name: str,
    dataset_version: str,
    source_case_id: str,
    failure_layer: str,
) -> str:
    raw_id = f"bad_{dataset_name}_{dataset_version}_{source_case_id}_{failure_layer}"
    return _slugify(raw_id)


def build_regression_case_draft(
    record: BadCaseRecord,
    *,
    target_dataset_name: str | None = None,
) -> RegressionCaseDraft:
    dataset_name = target_dataset_name or record.regression_dataset_name
    if dataset_name is None:
        raise ValueError("target dataset name is required to build regression case draft")

    source_case_id = record.source_case_id or record.id
    suggested_case_id = _slugify(f"{source_case_id}_regression_{record.failure_layer}")

    return RegressionCaseDraft(
        source_bad_case_id=record.id,
        target_dataset_name=dataset_name,
        suggested_case_id=suggested_case_id,
        title=f"Regression for {record.title}",
        input_summary=record.evidence_summary,
        expected_behavior=record.expected_behavior,
        assertions=[
            f"Expected behavior: {record.expected_behavior}",
            f"Actual bad behavior to avoid: {record.actual_behavior}",
            f"Failure layer: {record.failure_layer}",
            f"Regression action: {record.regression_action}",
        ],
        tags=_unique_strings(
            [
                "regression",
                "from_bad_case",
                record.failure_layer,
                record.failure_category,
                record.severity,
            ]
        ),
    )


def mark_bad_case_regression_added(
    record: BadCaseRecord,
    *,
    regression_case_id: str,
    regression_dataset_name: str | None = None,
) -> BadCaseRecord:
    return record.model_copy(
        update={
            "status": "regression_added",
            "regression_case_id": regression_case_id.strip(),
            "regression_dataset_name": (
                regression_dataset_name.strip()
                if isinstance(regression_dataset_name, str)
                else record.regression_dataset_name
            ),
        }
    )


def _map_analysis_category_to_failure_layer(category: str) -> BadCaseFailureLayer:
    mapping: dict[str, BadCaseFailureLayer] = {
        "intent_classification": "intent",
        "ticket_field_extraction": "field_extraction",
        "agent_routing": "routing",
        "rag_retrieval_or_citation": "rag_citation",
        "agent_decision_after_rag": "agent_decision",
    }
    return mapping.get(category, "unknown")


def _task_type_from_suite(suite_name: str) -> BadCaseTaskType:
    if suite_name == "rag":
        return "rag"
    if suite_name in {"intent", "field", "route"}:
        return "agent"
    return "unknown"


def _severity_from_priority(priority: str | None) -> BadCaseSeverity:
    if priority == "p0":
        return "critical"
    if priority == "p1":
        return "high"
    if priority == "p2":
        return "medium"
    return "medium"


def _extract_expected_behavior(lines: Sequence[str]) -> str:
    text = " ".join(lines)
    match = re.search(r"expected(?:_status|_path)?=([^ ]+)", text)
    if match:
        return match.group(1).strip()
    return "See original eval expectation."


def _extract_actual_behavior(lines: Sequence[str]) -> str:
    text = " ".join(lines)
    match = re.search(r"actual(?:_status|_path)?=([^ ]+)", text)
    if match:
        return match.group(1).strip()
    return "See original bad case evidence."


def _summarize_evidence(lines: Sequence[str]) -> str:
    compact = " ".join(line.strip() for line in lines if line.strip())
    if len(compact) <= 500:
        return compact
    return f"{compact[:497]}..."


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "bad_case"


def _unique_strings(values: Sequence[str]) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)
    return unique_values


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


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())
