import pytest

from app.core.release_control import (
    RollbackSignal,
    RolloutPolicy,
    assign_rollout_version,
    build_default_rollout_policies,
    build_rollback_decision,
    format_rollout_policies,
    stable_percentage_bucket,
    summarize_rollout_policies,
)


def test_default_rollout_policies_cover_model_rag_and_safety_canaries() -> None:
    policies = build_default_rollout_policies()
    summary = summarize_rollout_policies(policies)

    assert [policy.target for policy in policies] == [
        "model",
        "rag_parameters",
        "safety_policy",
    ]
    assert summary.policy_count == 3
    assert summary.target_counts == {
        "model": 1,
        "rag_parameters": 1,
        "safety_policy": 1,
    }
    assert summary.kill_switch_count == 3
    assert all(policy.feature_flags[0].kill_switch for policy in policies)
    assert all(policy.guardrail_metric_names for policy in policies)


def test_format_rollout_policies_is_readable() -> None:
    lines = format_rollout_policies(build_default_rollout_policies())

    assert lines[0] == "Release rollout policies"
    assert "policies: 3" in lines
    assert any("llm-balanced-model-canary" in line for line in lines)
    assert any("safety-policy-canary" in line for line in lines)


def test_rollout_policy_validates_status_and_percentage() -> None:
    with pytest.raises(ValueError, match="disabled rollout must use 0 percent"):
        RolloutPolicy(
            name="bad-disabled-policy",
            target="model",
            status="disabled",
            stable_version="stable",
            candidate_version="candidate",
            rollout_percentage=10,
            rollback_hint="Disable the canary.",
        )

    with pytest.raises(ValueError, match="full rollout must use 100 percent"):
        RolloutPolicy(
            name="bad-full-policy",
            target="model",
            status="full",
            stable_version="stable",
            candidate_version="candidate",
            rollout_percentage=99,
            rollback_hint="Disable the canary.",
        )


def test_assign_rollout_version_respects_disabled_and_internal_status() -> None:
    disabled = RolloutPolicy(
        name="model-canary",
        target="model",
        status="disabled",
        stable_version="stable-model",
        candidate_version="candidate-model",
        rollout_percentage=0,
        rollback_hint="Disable canary.",
    )
    internal = disabled.model_copy(
        update={
            "status": "internal",
            "enabled_tenant_tiers": ["internal"],
        }
    )

    disabled_assignment = assign_rollout_version(
        disabled,
        subject_id="user-001",
        tenant_tier="internal",
    )
    internal_assignment = assign_rollout_version(
        internal,
        subject_id="user-001",
        tenant_tier="internal",
    )

    assert disabled_assignment.selected_version == "stable-model"
    assert disabled_assignment.candidate_selected is False
    assert disabled_assignment.reason == "disabled"
    assert internal_assignment.selected_version == "candidate-model"
    assert internal_assignment.candidate_selected is True
    assert internal_assignment.reason == "internal_enabled"


def test_assign_rollout_version_respects_tenant_tier_and_canary_bucket() -> None:
    policy = RolloutPolicy(
        name="model-canary",
        target="model",
        status="canary",
        stable_version="stable-model",
        candidate_version="candidate-model",
        rollout_percentage=100,
        enabled_tenant_tiers=["beta"],
        rollback_hint="Disable canary.",
    )

    public_assignment = assign_rollout_version(
        policy,
        subject_id="user-001",
        tenant_tier="public",
    )
    beta_assignment = assign_rollout_version(
        policy,
        subject_id="user-001",
        tenant_tier="beta",
    )

    assert public_assignment.selected_version == "stable-model"
    assert public_assignment.reason == "tenant_tier_not_enabled"
    assert beta_assignment.selected_version == "candidate-model"
    assert beta_assignment.candidate_selected is True


def test_stable_percentage_bucket_is_deterministic_and_bounded() -> None:
    first = stable_percentage_bucket("model-canary:user-001")
    second = stable_percentage_bucket("model-canary:user-001")

    assert first == second
    assert 0 <= first < 100


def test_build_rollback_decision_rolls_back_on_critical_signal() -> None:
    policy = _canary_policy()
    signal = RollbackSignal(
        metric_name="llm.failures",
        current_value=0.2,
        threshold=0.1,
        severity="critical",
        message="LLM failure rate breached critical threshold.",
    )

    decision = build_rollback_decision(policy, signals=[signal])

    assert decision.action == "rollback"
    assert decision.should_rollback is True
    assert decision.reason == "critical_guardrail_breached"
    assert decision.blocking_signals == [signal]
    assert decision.stable_version == "stable-model"
    assert decision.candidate_version == "candidate-model"


def test_build_rollback_decision_holds_on_warning_signals_and_continues_when_clean() -> None:
    policy = _canary_policy()
    warning = RollbackSignal(
        metric_name="http.server.duration",
        current_value=5200,
        threshold=5000,
        severity="warning",
        message="p95 latency is high.",
    )

    hold = build_rollback_decision(policy, signals=[warning])
    clean = build_rollback_decision(policy, signals=[])

    assert hold.action == "hold"
    assert hold.reason == "warning_guardrail_breached"
    assert clean.action == "continue"
    assert clean.reason == "guardrails_within_range"


def _canary_policy() -> RolloutPolicy:
    return RolloutPolicy(
        name="model-canary",
        target="model",
        status="canary",
        stable_version="stable-model",
        candidate_version="candidate-model",
        rollout_percentage=10,
        enabled_tenant_tiers=["beta"],
        guardrail_metric_names=["llm.failures", "http.server.duration"],
        rollback_hint="Disable model canary.",
    )
