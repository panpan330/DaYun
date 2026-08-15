from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RolloutTarget = Literal[
    "model",
    "prompt",
    "rag_parameters",
    "routing_policy",
    "safety_policy",
    "feature",
]
RolloutStatus = Literal["disabled", "internal", "canary", "expanded", "full", "rolled_back"]
RollbackSeverity = Literal["critical", "warning"]
RollbackDecisionAction = Literal["continue", "hold", "rollback"]


class FeatureFlagSpec(BaseModel):
    name: str = Field(min_length=1)
    enabled: bool = False
    description: str = Field(min_length=1)
    owner: str = Field(default="ai-platform")
    kill_switch: bool = False

    @field_validator("name", "description", "owner", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RolloutPolicy(BaseModel):
    name: str = Field(min_length=1)
    target: RolloutTarget
    status: RolloutStatus = "disabled"
    stable_version: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    rollout_percentage: int = Field(default=0, ge=0, le=100)
    enabled_tenant_tiers: list[str] = Field(default_factory=list)
    feature_flags: list[FeatureFlagSpec] = Field(default_factory=list)
    guardrail_metric_names: list[str] = Field(default_factory=list)
    rollback_hint: str = Field(min_length=1)

    @field_validator(
        "name",
        "stable_version",
        "candidate_version",
        "rollback_hint",
        mode="before",
    )
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("enabled_tenant_tiers", "guardrail_metric_names", mode="before")
    @classmethod
    def normalize_string_lists(cls, value: object) -> object:
        return _normalize_string_list(value, field_name="rollout list")

    @model_validator(mode="after")
    def validate_status_and_percentage(self) -> "RolloutPolicy":
        if self.status == "disabled" and self.rollout_percentage != 0:
            raise ValueError("disabled rollout must use 0 percent")
        if self.status == "full" and self.rollout_percentage != 100:
            raise ValueError("full rollout must use 100 percent")
        if self.status == "rolled_back" and self.rollout_percentage != 0:
            raise ValueError("rolled_back rollout must use 0 percent")
        return self


class RolloutAssignment(BaseModel):
    policy_name: str = Field(min_length=1)
    selected_version: str = Field(min_length=1)
    candidate_selected: bool
    reason: str = Field(min_length=1)
    rollout_percentage: int = Field(ge=0, le=100)


class RollbackSignal(BaseModel):
    metric_name: str = Field(min_length=1)
    current_value: float
    threshold: float
    severity: RollbackSeverity
    message: str = Field(min_length=1)

    @field_validator("metric_name", "message", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RollbackDecision(BaseModel):
    action: RollbackDecisionAction
    reason: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    stable_version: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    blocking_signals: list[RollbackSignal] = Field(default_factory=list)

    @property
    def should_rollback(self) -> bool:
        return self.action == "rollback"


class RolloutCatalogSummary(BaseModel):
    policy_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    target_counts: dict[str, int] = Field(default_factory=dict)
    kill_switch_count: int = Field(ge=0)


def build_default_rollout_policies() -> list[RolloutPolicy]:
    return [
        RolloutPolicy(
            name="llm-balanced-model-canary",
            target="model",
            status="internal",
            stable_version="qwen3.7-max",
            candidate_version="next-balanced-model",
            rollout_percentage=0,
            enabled_tenant_tiers=["internal"],
            feature_flags=[
                FeatureFlagSpec(
                    name="llm_model_canary_enabled",
                    enabled=False,
                    description="Allow internal traffic to test the candidate balanced model.",
                    kill_switch=True,
                )
            ],
            guardrail_metric_names=[
                "llm.failures",
                "http.server.duration",
                "llm.estimated_cost",
            ],
            rollback_hint="Disable llm_model_canary_enabled or restore stable_version in model routing config.",
        ),
        RolloutPolicy(
            name="rag-parameter-canary",
            target="rag_parameters",
            status="disabled",
            stable_version="rag-params-stage9-v1",
            candidate_version="rag-params-next",
            rollout_percentage=0,
            enabled_tenant_tiers=["internal"],
            feature_flags=[
                FeatureFlagSpec(
                    name="rag_parameter_canary_enabled",
                    enabled=False,
                    description="Test candidate top_k, score_threshold, rerank, or compression settings.",
                    kill_switch=True,
                )
            ],
            guardrail_metric_names=[
                "rag.retrieval.empty_results",
                "rag.citation.failures",
                "http.server.duration",
            ],
            rollback_hint="Disable RAG canary flag and restore stable retrieval/rerank parameters.",
        ),
        RolloutPolicy(
            name="safety-policy-canary",
            target="safety_policy",
            status="disabled",
            stable_version="safety-stage10-v1",
            candidate_version="safety-next",
            rollout_percentage=0,
            enabled_tenant_tiers=["internal"],
            feature_flags=[
                FeatureFlagSpec(
                    name="safety_policy_canary_enabled",
                    enabled=False,
                    description="Test candidate prompt-injection or privacy protection rules.",
                    kill_switch=True,
                )
            ],
            guardrail_metric_names=[
                "safety.blocks",
                "http.server.requests",
            ],
            rollback_hint="Disable safety canary flag and restore stable safety rules.",
        ),
    ]


def assign_rollout_version(
    policy: RolloutPolicy,
    *,
    subject_id: str,
    tenant_tier: str | None = None,
) -> RolloutAssignment:
    if policy.status in {"disabled", "rolled_back"}:
        return _assignment(policy, selected=policy.stable_version, reason=policy.status)

    normalized_tier = (tenant_tier or "").strip()
    if policy.enabled_tenant_tiers and normalized_tier not in policy.enabled_tenant_tiers:
        return _assignment(policy, selected=policy.stable_version, reason="tenant_tier_not_enabled")

    if policy.status == "internal":
        return _assignment(policy, selected=policy.candidate_version, reason="internal_enabled")

    if policy.status == "full":
        return _assignment(policy, selected=policy.candidate_version, reason="full_rollout")

    bucket = stable_percentage_bucket(f"{policy.name}:{subject_id}")
    if bucket < policy.rollout_percentage:
        return _assignment(policy, selected=policy.candidate_version, reason=f"bucket_{bucket}_in_canary")
    return _assignment(policy, selected=policy.stable_version, reason=f"bucket_{bucket}_outside_canary")


def stable_percentage_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def build_rollback_decision(
    policy: RolloutPolicy,
    *,
    signals: Sequence[RollbackSignal],
) -> RollbackDecision:
    critical_signals = [signal for signal in signals if signal.severity == "critical"]
    warning_signals = [signal for signal in signals if signal.severity == "warning"]

    if critical_signals:
        return _rollback_decision(
            policy,
            action="rollback",
            reason="critical_guardrail_breached",
            signals=critical_signals,
        )
    if len(warning_signals) >= 2:
        return _rollback_decision(
            policy,
            action="hold",
            reason="multiple_warning_guardrails_breached",
            signals=warning_signals,
        )
    if warning_signals:
        return _rollback_decision(
            policy,
            action="hold",
            reason="warning_guardrail_breached",
            signals=warning_signals,
        )
    return _rollback_decision(
        policy,
        action="continue",
        reason="guardrails_within_range",
        signals=[],
    )


def summarize_rollout_policies(
    policies: Sequence[RolloutPolicy],
) -> RolloutCatalogSummary:
    status_counts = Counter(policy.status for policy in policies)
    target_counts = Counter(policy.target for policy in policies)
    kill_switch_count = sum(
        1
        for policy in policies
        for flag in policy.feature_flags
        if flag.kill_switch
    )
    return RolloutCatalogSummary(
        policy_count=len(policies),
        status_counts=dict(sorted(status_counts.items())),
        target_counts=dict(sorted(target_counts.items())),
        kill_switch_count=kill_switch_count,
    )


def format_rollout_policies(policies: Sequence[RolloutPolicy]) -> list[str]:
    summary = summarize_rollout_policies(policies)
    lines = [
        "Release rollout policies",
        f"policies: {summary.policy_count}",
        f"statuses: {_format_counts(summary.status_counts)}",
        f"targets: {_format_counts(summary.target_counts)}",
        f"kill_switches: {summary.kill_switch_count}",
    ]
    for policy in policies:
        flags = ", ".join(flag.name for flag in policy.feature_flags) or "-"
        lines.append(
            (
                f"- {policy.name}: target={policy.target} status={policy.status} "
                f"stable={policy.stable_version} candidate={policy.candidate_version} "
                f"rollout={policy.rollout_percentage}% flags={flags}"
            )
        )
    return lines


def _assignment(policy: RolloutPolicy, *, selected: str, reason: str) -> RolloutAssignment:
    return RolloutAssignment(
        policy_name=policy.name,
        selected_version=selected,
        candidate_selected=selected == policy.candidate_version,
        reason=reason,
        rollout_percentage=policy.rollout_percentage,
    )


def _rollback_decision(
    policy: RolloutPolicy,
    *,
    action: RollbackDecisionAction,
    reason: str,
    signals: Sequence[RollbackSignal],
) -> RollbackDecision:
    return RollbackDecision(
        action=action,
        reason=reason,
        policy_name=policy.name,
        stable_version=policy.stable_version,
        candidate_version=policy.candidate_version,
        blocking_signals=list(signals),
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
