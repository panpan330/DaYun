import json

from app.evaluation.bad_case_registry import BadCaseRecord
from app.evaluation.production_bad_case_registry import append_production_bad_case


def _record() -> BadCaseRecord:
    return BadCaseRecord(
        id="bad_production_feedback_stage11_feedback_1_agent_decision",
        title="Production feedback 1: incorrect decision",
        source="production",
        task_type="agent",
        severity="medium",
        status="regression_added",
        source_case_id="feedback_1",
        failure_layer="agent_decision",
        failure_category="incorrect decision",
        expected_behavior="Give the correct escalation path.",
        actual_behavior="The answer did not offer escalation.",
        recommended_action="Adjust the decision policy.",
        regression_action="Add a regression case.",
        regression_dataset_name="agent_eval",
        regression_case_id="feedback_1_regression_agent_decision",
        evidence_summary="feedback_id=1",
        tags=["production_feedback"],
    )


def test_append_production_bad_case_is_idempotent_and_writes_valid_registry(tmp_path) -> None:
    path = tmp_path / "bad_cases.json"
    path.write_text('{"schema_version":"stage10.bad_case_registry.v1","records":[]}\n', encoding="utf-8")

    first = append_production_bad_case(path, _record())
    second = append_production_bad_case(path, _record())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert first.id == second.id
    assert [item["id"] for item in payload["records"]] == [first.id]
