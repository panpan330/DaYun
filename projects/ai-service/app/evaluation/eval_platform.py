from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.eval_suite import AgentEvalRunReport


EvalTaskType = Literal[
    "agent",
    "rag_retrieval",
    "rag_answer",
    "tool_calling",
    "safety",
]
EvalMetricDirection = Literal["higher_is_better", "lower_is_better"]


class EvalDatasetManifest(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task_type: EvalTaskType
    cases_path: str = Field(min_length=1)
    description: str = ""
    frozen: bool = True
    baseline_run_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("name", "version", "cases_path", "description", "baseline_run_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="tags")


class EvalDatasetRegistry(BaseModel):
    registry_version: str = Field(min_length=1)
    datasets: list[EvalDatasetManifest] = Field(default_factory=list)

    @field_validator("registry_version", mode="before")
    @classmethod
    def normalize_registry_version(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_unique_dataset_versions(self) -> "EvalDatasetRegistry":
        seen: set[tuple[str, str]] = set()
        for dataset in self.datasets:
            key = (dataset.name, dataset.version)
            if key in seen:
                raise ValueError("eval dataset registry contains duplicate name/version")
            seen.add(key)
        return self


class EvalRunContext(BaseModel):
    run_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    baseline_run_id: str | None = None
    model_name: str = "fake_or_rule_based"
    prompt_version: str = "not_applicable"
    code_version: str = "local"
    evaluator_version: str = "eval-platform-v1"
    notes: str = ""
    started_at: datetime | None = None

    @field_validator(
        "run_id",
        "dataset_name",
        "dataset_version",
        "candidate_version",
        "baseline_run_id",
        "model_name",
        "prompt_version",
        "code_version",
        "evaluator_version",
        "notes",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return value


class EvalMetric(BaseModel):
    name: str = Field(min_length=1)
    value: float
    direction: EvalMetricDirection = "higher_is_better"

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class EvalRunSnapshot(BaseModel):
    context: EvalRunContext
    evaluated_check_count: int = Field(ge=0)
    passed_check_count: int = Field(ge=0)
    failed_check_count: int = Field(ge=0)
    passed: bool
    metrics: list[EvalMetric] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_check_counts(self) -> "EvalRunSnapshot":
        if self.passed_check_count + self.failed_check_count != self.evaluated_check_count:
            raise ValueError("passed_check_count + failed_check_count must equal evaluated_check_count")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("eval run snapshot metric names must be unique")
        return self

    def metric_map(self) -> dict[str, EvalMetric]:
        return {metric.name: metric for metric in self.metrics}


class EvalMetricComparison(BaseModel):
    name: str = Field(min_length=1)
    baseline_value: float
    candidate_value: float
    delta: float
    direction: EvalMetricDirection
    regressed: bool


class EvalRegressionReport(BaseModel):
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    baseline_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)
    baseline_candidate_version: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    regressed: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    metric_comparisons: list[EvalMetricComparison] = Field(default_factory=list)


def load_eval_dataset_registry(path: Path | str) -> EvalDatasetRegistry:
    raw_data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalDatasetRegistry.model_validate(raw_data)


def find_eval_dataset_manifest(
    registry: EvalDatasetRegistry,
    *,
    name: str,
    version: str | None = None,
) -> EvalDatasetManifest:
    candidates = [
        dataset
        for dataset in registry.datasets
        if dataset.name == name and (version is None or dataset.version == version)
    ]
    if not candidates:
        requested = f"{name}:{version}" if version is not None else name
        raise ValueError(f"Unknown eval dataset: {requested}")
    if version is None and len(candidates) > 1:
        versions = ", ".join(dataset.version for dataset in candidates)
        raise ValueError(f"Eval dataset {name} has multiple versions: {versions}")
    return candidates[0]


def format_eval_dataset_registry(registry: EvalDatasetRegistry) -> list[str]:
    lines = [
        "Evaluation dataset registry",
        f"registry_version: {registry.registry_version}",
        f"datasets: {len(registry.datasets)}",
    ]
    for dataset in registry.datasets:
        baseline = dataset.baseline_run_id or "-"
        frozen = str(dataset.frozen).lower()
        tags = ", ".join(dataset.tags) if dataset.tags else "-"
        lines.append(
            (
                f"- {dataset.name}:{dataset.version} "
                f"type={dataset.task_type} frozen={frozen} "
                f"baseline={baseline} tags={tags} path={dataset.cases_path}"
            )
        )
    return lines


def build_agent_eval_run_snapshot(
    report: AgentEvalRunReport,
    *,
    context: EvalRunContext,
) -> EvalRunSnapshot:
    if context.started_at is None:
        context = context.model_copy(update={"started_at": datetime.now(timezone.utc)})
    evaluated_checks = sum(suite.case_count for suite in report.suite_reports)
    failed_checks = sum(suite.failed_case_count for suite in report.suite_reports)
    passed_checks = evaluated_checks - failed_checks
    suite_pass_rate = _ratio(report.passed_suite_count, report.suite_count, default=1.0)
    check_pass_rate = _ratio(passed_checks, evaluated_checks, default=1.0)

    return EvalRunSnapshot(
        context=context,
        evaluated_check_count=evaluated_checks,
        passed_check_count=passed_checks,
        failed_check_count=failed_checks,
        passed=report.passed,
        metrics=[
            EvalMetric(name="suite_pass_rate", value=suite_pass_rate),
            EvalMetric(name="check_pass_rate", value=check_pass_rate),
            EvalMetric(
                name="failed_suites",
                value=float(report.failed_suite_count),
                direction="lower_is_better",
            ),
            EvalMetric(
                name="failed_checks",
                value=float(failed_checks),
                direction="lower_is_better",
            ),
        ],
    )


def compare_eval_run_snapshots(
    baseline: EvalRunSnapshot,
    candidate: EvalRunSnapshot,
    *,
    tolerance: float = 0.0,
) -> EvalRegressionReport:
    if baseline.context.dataset_name != candidate.context.dataset_name:
        raise ValueError("Cannot compare eval runs from different datasets")
    if baseline.context.dataset_version != candidate.context.dataset_version:
        raise ValueError("Cannot compare eval runs from different dataset versions")
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    baseline_metrics = baseline.metric_map()
    candidate_metrics = candidate.metric_map()
    common_metric_names = sorted(set(baseline_metrics) & set(candidate_metrics))
    comparisons = [
        _compare_metric(
            baseline_metrics[name],
            candidate_metrics[name],
            tolerance=tolerance,
        )
        for name in common_metric_names
    ]
    blocking_reasons = _build_regression_blocking_reasons(
        baseline=baseline,
        candidate=candidate,
        comparisons=comparisons,
    )

    return EvalRegressionReport(
        dataset_name=baseline.context.dataset_name,
        dataset_version=baseline.context.dataset_version,
        baseline_run_id=baseline.context.run_id,
        candidate_run_id=candidate.context.run_id,
        baseline_candidate_version=baseline.context.candidate_version,
        candidate_version=candidate.context.candidate_version,
        regressed=bool(blocking_reasons),
        blocking_reasons=blocking_reasons,
        metric_comparisons=comparisons,
    )


def format_eval_regression_report(report: EvalRegressionReport) -> list[str]:
    lines = [
        "Evaluation regression report",
        f"dataset: {report.dataset_name}:{report.dataset_version}",
        f"baseline_run_id: {report.baseline_run_id}",
        f"candidate_run_id: {report.candidate_run_id}",
        f"baseline_candidate_version: {report.baseline_candidate_version}",
        f"candidate_version: {report.candidate_version}",
        f"regressed: {str(report.regressed).lower()}",
    ]
    if report.blocking_reasons:
        lines.append("blocking_reasons:")
        lines.extend(f"- {reason}" for reason in report.blocking_reasons)
    lines.append("metrics:")
    for comparison in report.metric_comparisons:
        lines.append(
            (
                f"- {comparison.name}: baseline={comparison.baseline_value:.6f} "
                f"candidate={comparison.candidate_value:.6f} "
                f"delta={comparison.delta:.6f} "
                f"direction={comparison.direction} "
                f"regressed={str(comparison.regressed).lower()}"
            )
        )
    return lines


def _compare_metric(
    baseline_metric: EvalMetric,
    candidate_metric: EvalMetric,
    *,
    tolerance: float,
) -> EvalMetricComparison:
    if baseline_metric.direction != candidate_metric.direction:
        raise ValueError(f"Metric direction mismatch for {baseline_metric.name}")
    delta = round(candidate_metric.value - baseline_metric.value, 6)
    if baseline_metric.direction == "higher_is_better":
        regressed = candidate_metric.value < baseline_metric.value - tolerance
    else:
        regressed = candidate_metric.value > baseline_metric.value + tolerance
    return EvalMetricComparison(
        name=baseline_metric.name,
        baseline_value=baseline_metric.value,
        candidate_value=candidate_metric.value,
        delta=delta,
        direction=baseline_metric.direction,
        regressed=regressed,
    )


def _build_regression_blocking_reasons(
    *,
    baseline: EvalRunSnapshot,
    candidate: EvalRunSnapshot,
    comparisons: Sequence[EvalMetricComparison],
) -> list[str]:
    reasons = [
        f"metric_regressed:{comparison.name}"
        for comparison in comparisons
        if comparison.regressed
    ]
    if baseline.passed and not candidate.passed:
        reasons.append("overall_status_regressed")
    if candidate.failed_check_count > baseline.failed_check_count:
        reasons.append("failed_check_count_increased")
    return sorted(set(reasons))


def _ratio(numerator: int, denominator: int, *, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return round(numerator / denominator, 6)


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
