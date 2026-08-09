import pytest
from pydantic import ValidationError

from app.core.slo_runbook import (
    IncidentResponsePlan,
    IncidentSignal,
    RunbookSpec,
    RunbookStep,
    build_default_incident_response_plan,
    build_ratio_error_budget_snapshot,
    classify_incident,
    format_incident_response_plan,
)


def test_default_incident_response_plan_contains_ai_slos_and_runbooks() -> None:
    plan = build_default_incident_response_plan()

    objective_names = {objective.name for objective in plan.objectives}
    runbook_names = {runbook.name for runbook in plan.runbooks}

    assert plan.service_name == "ai-service"
    assert "ai_service_availability" in objective_names
    assert "llm_success_ratio" in objective_names
    assert "llm_provider_failure_runbook" in runbook_names
    assert "high_latency_runbook" in runbook_names


def test_ratio_error_budget_snapshot_marks_exhausted_when_actual_below_budget() -> None:
    plan = build_default_incident_response_plan()
    objective = next(item for item in plan.objectives if item.name == "ai_service_availability")

    snapshot = build_ratio_error_budget_snapshot(objective, actual_ratio=0.992)

    assert snapshot.target_ratio == 0.995
    assert snapshot.allowed_bad_ratio == pytest.approx(0.005)
    assert snapshot.consumed_bad_ratio == pytest.approx(0.008)
    assert snapshot.consumed_percentage == 160
    assert snapshot.remaining_percentage == 0
    assert snapshot.exhausted is True


def test_classify_incident_uses_highest_breached_severity() -> None:
    classification = classify_incident(
        [
            IncidentSignal(
                metric_name="http.server.duration",
                severity="p2",
                current_value=6500,
                comparator=">=",
                threshold=5000,
                message="p95 latency is high",
            ),
            IncidentSignal(
                metric_name="llm.failures",
                severity="p1",
                current_value=0.12,
                comparator=">=",
                threshold=0.1,
                message="LLM failure rate is high",
            ),
        ]
    )

    assert classification.severity == "p1"
    assert classification.action == "rollback"
    assert classification.reason == "p1_signal_breached"
    assert [signal.metric_name for signal in classification.breached_signals] == [
        "llm.failures",
        "http.server.duration",
    ]


def test_classify_incident_returns_monitor_when_nothing_breached() -> None:
    classification = classify_incident(
        [
            IncidentSignal(
                metric_name="llm.failures",
                severity="p1",
                current_value=0.01,
                comparator=">=",
                threshold=0.1,
                message="LLM failure rate is normal",
            )
        ]
    )

    assert classification.severity is None
    assert classification.action == "monitor"
    assert classification.breached_signals == []


def test_runbook_rejects_unsorted_duplicate_step_order() -> None:
    with pytest.raises(ValidationError, match="runbook step order must be unique"):
        RunbookSpec(
            name="bad_runbook",
            trigger_alerts=["High LLM failure rate"],
            severity="p1",
            goal="Fix outage",
            owner="ai-platform",
            escalation_hint="Escalate to owner",
            steps=[
                RunbookStep(
                    order=1,
                    title="First",
                    action="investigate",
                    instruction="Check signal",
                    expected_signal="Signal is identified",
                ),
                RunbookStep(
                    order=1,
                    title="Duplicate",
                    action="rollback",
                    instruction="Rollback",
                    expected_signal="Service recovers",
                ),
            ],
        )


def test_incident_response_plan_rejects_duplicate_objective_names() -> None:
    plan = build_default_incident_response_plan()

    with pytest.raises(ValidationError, match="SLO names must be unique"):
        IncidentResponsePlan(
            service_name=plan.service_name,
            objectives=[plan.objectives[0], plan.objectives[0]],
            runbooks=plan.runbooks,
        )


def test_format_incident_response_plan_contains_summary_lines() -> None:
    lines = format_incident_response_plan(build_default_incident_response_plan())

    assert lines[0] == "Incident response plan: ai-service"
    assert any("ai_service_availability" in line for line in lines)
    assert any("llm_provider_failure_runbook" in line for line in lines)
