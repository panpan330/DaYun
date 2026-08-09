from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.production_monitoring import AlertComparator, evaluate_alert_condition


ServiceObjectiveDomain = Literal[
    "availability",
    "latency",
    "error_rate",
    "cost",
    "rag_quality",
    "tooling",
    "safety",
]
ServiceObjectiveWindow = Literal["7d", "30d", "90d"]
IncidentSeverity = Literal["p0", "p1", "p2", "p3"]
IncidentAction = Literal["investigate", "degrade", "rollback", "escalate", "monitor"]

_SEVERITY_RANK: dict[IncidentSeverity, int] = {
    "p0": 0,
    "p1": 1,
    "p2": 2,
    "p3": 3,
}


class ServiceLevelIndicatorSpec(BaseModel):
    name: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    good_event_description: str = Field(min_length=1)
    total_event_description: str = Field(min_length=1)
    unit: str = Field(min_length=1)

    @field_validator(
        "name",
        "metric_name",
        "description",
        "good_event_description",
        "total_event_description",
        "unit",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ServiceLevelObjectiveSpec(BaseModel):
    name: str = Field(min_length=1)
    domain: ServiceObjectiveDomain
    sli: ServiceLevelIndicatorSpec
    comparator: AlertComparator
    target_value: float = Field(ge=0)
    window: ServiceObjectiveWindow
    owner: str = Field(min_length=1)
    user_impact: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @field_validator("name", "owner", "user_impact", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_ratio_targets(self) -> "ServiceLevelObjectiveSpec":
        if self.sli.unit == "ratio" and self.target_value > 1:
            raise ValueError("ratio SLO target_value must be between 0 and 1")
        return self


class ErrorBudgetSnapshot(BaseModel):
    objective_name: str = Field(min_length=1)
    target_ratio: float = Field(ge=0, le=1)
    actual_ratio: float = Field(ge=0, le=1)
    allowed_bad_ratio: float = Field(ge=0, le=1)
    consumed_bad_ratio: float = Field(ge=0, le=1)
    consumed_percentage: float = Field(ge=0)
    remaining_percentage: float
    exhausted: bool


class IncidentSignal(BaseModel):
    metric_name: str = Field(min_length=1)
    severity: IncidentSeverity
    current_value: float
    comparator: AlertComparator
    threshold: float
    message: str = Field(min_length=1)

    @field_validator("metric_name", "message", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @property
    def breached(self) -> bool:
        return evaluate_alert_condition(
            value=self.current_value,
            comparator=self.comparator,
            threshold=self.threshold,
        )


class IncidentClassification(BaseModel):
    severity: IncidentSeverity | None
    action: IncidentAction
    reason: str = Field(min_length=1)
    breached_signals: list[IncidentSignal] = Field(default_factory=list)


class RunbookStep(BaseModel):
    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    action: IncidentAction
    instruction: str = Field(min_length=1)
    expected_signal: str = Field(min_length=1)
    stop_condition: str | None = None

    @field_validator("title", "instruction", "expected_signal", "stop_condition", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class RunbookSpec(BaseModel):
    name: str = Field(min_length=1)
    trigger_alerts: list[str] = Field(default_factory=list)
    severity: IncidentSeverity
    goal: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    steps: list[RunbookStep] = Field(min_length=1)
    related_rollout_policies: list[str] = Field(default_factory=list)
    escalation_hint: str = Field(min_length=1)

    @field_validator("name", "goal", "owner", "escalation_hint", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("trigger_alerts", "related_rollout_policies", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="runbook list")

    @model_validator(mode="after")
    def validate_step_order(self) -> "RunbookSpec":
        orders = [step.order for step in self.steps]
        if len(orders) != len(set(orders)):
            raise ValueError("runbook step order must be unique")
        if orders != sorted(orders):
            raise ValueError("runbook steps must be sorted by order")
        return self


class IncidentResponsePlan(BaseModel):
    service_name: str = Field(min_length=1)
    objectives: list[ServiceLevelObjectiveSpec] = Field(min_length=1)
    runbooks: list[RunbookSpec] = Field(min_length=1)

    @field_validator("service_name", mode="before")
    @classmethod
    def normalize_service_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_unique_names(self) -> "IncidentResponsePlan":
        objective_names = [objective.name for objective in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("SLO names must be unique")
        runbook_names = [runbook.name for runbook in self.runbooks]
        if len(runbook_names) != len(set(runbook_names)):
            raise ValueError("runbook names must be unique")
        return self


class IncidentResponseSummary(BaseModel):
    service_name: str
    objective_count: int = Field(ge=0)
    runbook_count: int = Field(ge=0)
    objective_domain_counts: dict[str, int] = Field(default_factory=dict)
    runbook_severity_counts: dict[str, int] = Field(default_factory=dict)


def build_default_incident_response_plan() -> IncidentResponsePlan:
    objectives = [
        ServiceLevelObjectiveSpec(
            name="ai_service_availability",
            domain="availability",
            sli=ServiceLevelIndicatorSpec(
                name="successful_http_request_ratio",
                metric_name="http.server.requests",
                unit="ratio",
                description="Ratio of successful HTTP requests for AI user-facing routes.",
                good_event_description="HTTP requests that do not end in 5xx or infrastructure timeout.",
                total_event_description="All HTTP requests to user-facing AI routes.",
            ),
            comparator=">=",
            target_value=0.995,
            window="30d",
            owner="ai-platform",
            user_impact="Users cannot reliably use chat, RAG, or agent workflows.",
            description="The AI service should remain available for normal user-facing requests.",
        ),
        ServiceLevelObjectiveSpec(
            name="chat_p95_latency",
            domain="latency",
            sli=ServiceLevelIndicatorSpec(
                name="chat_http_p95_latency",
                metric_name="http.server.duration",
                unit="ms",
                description="P95 latency for chat and agent HTTP requests.",
                good_event_description="Requests that complete within the latency target.",
                total_event_description="All completed chat and agent requests.",
            ),
            comparator="<=",
            target_value=5000,
            window="30d",
            owner="ai-platform",
            user_impact="Users feel the assistant is slow or repeatedly interrupted.",
            description="Most AI requests should finish within a predictable latency boundary.",
        ),
        ServiceLevelObjectiveSpec(
            name="llm_success_ratio",
            domain="error_rate",
            sli=ServiceLevelIndicatorSpec(
                name="llm_success_ratio",
                metric_name="llm.failures",
                unit="ratio",
                description="Ratio of LLM calls that complete without provider, timeout, or auth errors.",
                good_event_description="LLM calls that return a valid model response.",
                total_event_description="All LLM calls attempted by AI service workflows.",
            ),
            comparator=">=",
            target_value=0.98,
            window="30d",
            owner="ai-platform",
            user_impact="Users receive fallback answers, empty answers, or failed requests.",
            description="LLM dependency health should stay high enough for normal AI workflows.",
        ),
        ServiceLevelObjectiveSpec(
            name="rag_citation_quality",
            domain="rag_quality",
            sli=ServiceLevelIndicatorSpec(
                name="rag_citation_pass_ratio",
                metric_name="rag.citation.failures",
                unit="ratio",
                description="Ratio of RAG answers whose citations pass source verification.",
                good_event_description="Answers with citations that can be matched back to retrieved context.",
                total_event_description="All RAG answers that claim external knowledge.",
            ),
            comparator=">=",
            target_value=0.97,
            window="30d",
            owner="ai-platform",
            user_impact="Users may receive unsupported answers or misleading source references.",
            description="RAG answers should preserve enough source grounding for customer-service use.",
        ),
        ServiceLevelObjectiveSpec(
            name="tool_success_ratio",
            domain="tooling",
            sli=ServiceLevelIndicatorSpec(
                name="tool_success_ratio",
                metric_name="tool.failures",
                unit="ratio",
                description="Ratio of tool executions that complete with validated business results.",
                good_event_description="Tool executions that pass validation and return usable results.",
                total_event_description="All tool executions requested by controlled AI workflows.",
            ),
            comparator=">=",
            target_value=0.98,
            window="30d",
            owner="ai-platform",
            user_impact="Users cannot query orders or create tickets through the AI workflow.",
            description="Tool execution should remain reliable after validation, auth, and Java service calls.",
        ),
    ]

    runbooks = [
        RunbookSpec(
            name="llm_provider_failure_runbook",
            trigger_alerts=["High LLM failure rate"],
            severity="p1",
            goal="Restore AI answer generation without leaking provider internals to users.",
            owner="ai-platform",
            related_rollout_policies=["llm-balanced-model-canary"],
            escalation_hint="Escalate to the owner of LLM provider configuration if auth, quota, or provider outage is suspected.",
            steps=[
                RunbookStep(
                    order=1,
                    title="Confirm scope",
                    action="investigate",
                    instruction="Check llm.failures by provider, model, operation, and error_code.",
                    expected_signal="A specific provider, model, error_code, or operation should explain most failures.",
                ),
                RunbookStep(
                    order=2,
                    title="Protect users",
                    action="degrade",
                    instruction="Keep safe fallback answers enabled and avoid exposing raw provider errors.",
                    expected_signal="User-facing errors stay stable and internal error detail stays in logs only.",
                ),
                RunbookStep(
                    order=3,
                    title="Rollback candidate model",
                    action="rollback",
                    instruction="If failures are tied to candidate model traffic, disable the canary flag.",
                    expected_signal="llm.failures drops after traffic returns to the stable model.",
                    stop_condition="Failure rate returns below alert threshold for one alert window.",
                ),
            ],
        ),
        RunbookSpec(
            name="high_latency_runbook",
            trigger_alerts=["High p95 request latency"],
            severity="p2",
            goal="Identify the slow stage and reduce user-visible waiting time.",
            owner="ai-platform",
            related_rollout_policies=["llm-balanced-model-canary", "rag-parameter-canary"],
            escalation_hint="Escalate to Java service, vector database, or LLM provider owner according to the slowest dependency.",
            steps=[
                RunbookStep(
                    order=1,
                    title="Break down latency",
                    action="investigate",
                    instruction="Compare http, llm, rag, rerank, tool, and java client timing.",
                    expected_signal="One or two stages should dominate the p95 latency increase.",
                ),
                RunbookStep(
                    order=2,
                    title="Reduce expensive path",
                    action="degrade",
                    instruction="Temporarily lower optional retrieval, rerank, fallback, or verbose generation paths.",
                    expected_signal="p95 latency decreases while core answer behavior remains usable.",
                ),
                RunbookStep(
                    order=3,
                    title="Hold rollout",
                    action="rollback",
                    instruction="If latency started after a canary, stop expansion or roll back the related policy.",
                    expected_signal="Latency trend moves back toward baseline after rollback.",
                ),
            ],
        ),
        RunbookSpec(
            name="rag_quality_drop_runbook",
            trigger_alerts=["RAG no-context spike"],
            severity="p2",
            goal="Recover grounded answers and avoid unsupported responses.",
            owner="ai-platform",
            related_rollout_policies=["rag-parameter-canary"],
            escalation_hint="Escalate to the knowledge-base owner if source documents, permissions, or ingestion changed.",
            steps=[
                RunbookStep(
                    order=1,
                    title="Check retrieval health",
                    action="investigate",
                    instruction="Inspect empty result rate, filters, score thresholds, collection status, and latest ingestion.",
                    expected_signal="The failure should map to data absence, filter mismatch, threshold issue, or vector store problem.",
                ),
                RunbookStep(
                    order=2,
                    title="Prefer safe no-answer",
                    action="degrade",
                    instruction="Keep no-context fallback strict instead of allowing the model to invent unsupported content.",
                    expected_signal="Unsupported answer rate stays controlled while retrieval is repaired.",
                ),
                RunbookStep(
                    order=3,
                    title="Restore stable RAG parameters",
                    action="rollback",
                    instruction="Disable candidate RAG parameters if the spike correlates with parameter canary traffic.",
                    expected_signal="RAG no-context and citation failures return near baseline.",
                ),
            ],
        ),
        RunbookSpec(
            name="cost_burn_runbook",
            trigger_alerts=["Cost burn rate high"],
            severity="p2",
            goal="Stop unexpected cost growth while preserving essential AI workflows.",
            owner="ai-platform",
            related_rollout_policies=["llm-balanced-model-canary"],
            escalation_hint="Escalate if traffic growth is legitimate and budget needs a product decision.",
            steps=[
                RunbookStep(
                    order=1,
                    title="Find cost driver",
                    action="investigate",
                    instruction="Group estimated cost by route, operation, model, fallback reason, and token usage.",
                    expected_signal="Cost growth should map to traffic, longer prompts, retries, fallback, or model tier.",
                ),
                RunbookStep(
                    order=2,
                    title="Apply cost guardrails",
                    action="degrade",
                    instruction="Tighten max output tokens, disable expensive fallback for low-priority traffic, or lower model tier.",
                    expected_signal="Hourly cost burn slows without breaking critical paths.",
                ),
                RunbookStep(
                    order=3,
                    title="Monitor recovery",
                    action="monitor",
                    instruction="Watch cost and error metrics for at least one full alert window.",
                    expected_signal="Cost returns below threshold and failure rate does not rise unexpectedly.",
                ),
            ],
        ),
    ]
    return IncidentResponsePlan(
        service_name="ai-service",
        objectives=objectives,
        runbooks=runbooks,
    )


def build_ratio_error_budget_snapshot(
    objective: ServiceLevelObjectiveSpec,
    *,
    actual_ratio: float,
) -> ErrorBudgetSnapshot:
    if objective.sli.unit != "ratio":
        raise ValueError("error budget calculation in this lesson only supports ratio SLOs")
    if objective.comparator != ">=":
        raise ValueError("ratio error budget requires a >= objective")
    if not 0 <= actual_ratio <= 1:
        raise ValueError("actual_ratio must be between 0 and 1")

    allowed_bad_ratio = 1 - objective.target_value
    consumed_bad_ratio = max(0.0, 1 - actual_ratio)
    consumed_percentage = (
        consumed_bad_ratio / allowed_bad_ratio * 100
        if allowed_bad_ratio > 0
        else 100.0
    )
    remaining_percentage = max(0.0, 100.0 - consumed_percentage)
    return ErrorBudgetSnapshot(
        objective_name=objective.name,
        target_ratio=objective.target_value,
        actual_ratio=actual_ratio,
        allowed_bad_ratio=allowed_bad_ratio,
        consumed_bad_ratio=consumed_bad_ratio,
        consumed_percentage=round(consumed_percentage, 2),
        remaining_percentage=round(remaining_percentage, 2),
        exhausted=consumed_percentage >= 100,
    )


def classify_incident(signals: Sequence[IncidentSignal]) -> IncidentClassification:
    breached = [signal for signal in signals if signal.breached]
    if not breached:
        return IncidentClassification(
            severity=None,
            action="monitor",
            reason="no_signal_breached",
            breached_signals=[],
        )

    highest = min(breached, key=lambda signal: _SEVERITY_RANK[signal.severity]).severity
    if highest == "p0":
        action: IncidentAction = "escalate"
    elif highest == "p1":
        action = "rollback"
    elif highest == "p2":
        action = "degrade"
    else:
        action = "investigate"

    return IncidentClassification(
        severity=highest,
        action=action,
        reason=f"{highest}_signal_breached",
        breached_signals=sorted(
            breached,
            key=lambda signal: _SEVERITY_RANK[signal.severity],
        ),
    )


def summarize_incident_response_plan(plan: IncidentResponsePlan) -> IncidentResponseSummary:
    objective_domain_counts = Counter(objective.domain for objective in plan.objectives)
    runbook_severity_counts = Counter(runbook.severity for runbook in plan.runbooks)
    return IncidentResponseSummary(
        service_name=plan.service_name,
        objective_count=len(plan.objectives),
        runbook_count=len(plan.runbooks),
        objective_domain_counts=dict(sorted(objective_domain_counts.items())),
        runbook_severity_counts=dict(sorted(runbook_severity_counts.items())),
    )


def format_incident_response_plan(plan: IncidentResponsePlan) -> list[str]:
    summary = summarize_incident_response_plan(plan)
    lines = [
        f"Incident response plan: {summary.service_name}",
        f"objectives: {summary.objective_count}",
        f"runbooks: {summary.runbook_count}",
        f"objective_domains: {_format_counts(summary.objective_domain_counts)}",
        f"runbook_severities: {_format_counts(summary.runbook_severity_counts)}",
        "",
        "SLOs:",
    ]
    lines.extend(
        (
            f"- {objective.name}: {objective.sli.name} "
            f"{objective.comparator} {objective.target_value} window={objective.window} "
            f"owner={objective.owner}"
        )
        for objective in plan.objectives
    )
    lines.extend(["", "Runbooks:"])
    lines.extend(
        (
            f"- {runbook.name}: severity={runbook.severity} "
            f"alerts={','.join(runbook.trigger_alerts) or '-'} steps={len(runbook.steps)}"
        )
        for runbook in plan.runbooks
    )
    return lines


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
