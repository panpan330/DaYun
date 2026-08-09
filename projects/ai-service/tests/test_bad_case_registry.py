import pytest

from app.agents.bad_case_analysis import BadCaseAnalysisItem
from app.evaluation.bad_case_registry import (
    BadCaseRecord,
    BadCaseRegistry,
    build_bad_case_record_from_analysis_item,
    build_bad_case_registry_summary,
    build_regression_case_draft,
    format_bad_case_registry_summary,
    mark_bad_case_regression_added,
)


def test_build_bad_case_record_from_analysis_item_preserves_eval_context() -> None:
    item = BadCaseAnalysisItem(
        suite_name="rag",
        suite_title="RAG + Agent evaluation",
        case_id="agent_policy_refund_arrival_001",
        priority="p0",
        category="rag_retrieval_or_citation",
        likely_layer="RAG retrieval or citation",
        diagnosis="The Agent did not cite the expected source.",
        recommended_action="Check retrieval and citation mapping.",
        regression_action="Keep this case in the RAG + Agent eval suite.",
        review_questions=["Does the knowledge base contain the expected source?"],
        evidence_lines=[
            "- agent_policy_refund_arrival_001: expected_status=answered actual_status=answered priority=p0",
            "  expected_sources: ['refund-return-policy.md']",
            "  actual_sources: ['account-security-faq.md']",
            "  - missing_sources=['refund-return-policy.md']",
        ],
    )

    record = build_bad_case_record_from_analysis_item(
        item,
        discovered_run_id="agent-eval-run-001",
        dataset_name="agent_eval",
        dataset_version="stage6-v1",
    )

    assert record.id == (
        "bad_agent_eval_stage6_v1_agent_policy_refund_arrival_001_rag_citation"
    )
    assert record.source == "eval"
    assert record.task_type == "rag"
    assert record.severity == "critical"
    assert record.status == "open"
    assert record.discovered_run_id == "agent-eval-run-001"
    assert record.dataset_name == "agent_eval"
    assert record.dataset_version == "stage6-v1"
    assert record.source_case_id == "agent_policy_refund_arrival_001"
    assert record.failure_layer == "rag_citation"
    assert record.failure_category == "rag_retrieval_or_citation"
    assert record.expected_behavior == "answered"
    assert record.actual_behavior == "answered"
    assert "missing_sources" in record.evidence_summary
    assert "regression_candidate" in record.tags
    assert "p0" in record.tags


def test_bad_case_registry_summary_counts_status_severity_and_layers() -> None:
    open_record = _record("case_001", status="open", severity="critical")
    regression_record = _record(
        "case_002",
        status="regression_added",
        severity="high",
        failure_layer="routing",
    )
    registry = BadCaseRegistry(
        schema_version="stage10.bad_case_registry.v1",
        records=[open_record, regression_record],
    )

    summary = build_bad_case_registry_summary(registry)
    lines = format_bad_case_registry_summary(summary)

    assert summary.record_count == 2
    assert summary.open_count == 1
    assert summary.regression_added_count == 1
    assert summary.severity_counts == {"critical": 1, "high": 1}
    assert summary.status_counts == {"open": 1, "regression_added": 1}
    assert summary.layer_counts == {"rag_citation": 1, "routing": 1}
    assert "records: 2" in lines
    assert "regression_added: 1" in lines


def test_bad_case_registry_rejects_duplicate_record_ids() -> None:
    with pytest.raises(ValueError, match="record ids must be unique"):
        BadCaseRegistry(
            schema_version="stage10.bad_case_registry.v1",
            records=[_record("case_001"), _record("case_001")],
        )


def test_build_regression_case_draft_from_bad_case_record() -> None:
    record = _record(
        "case_001",
        regression_dataset_name="agent_eval",
        source_case_id="agent_policy_refund_arrival_001",
        failure_layer="rag_citation",
    )

    draft = build_regression_case_draft(record)

    assert draft.source_bad_case_id == "case_001"
    assert draft.target_dataset_name == "agent_eval"
    assert draft.suggested_case_id == (
        "agent_policy_refund_arrival_001_regression_rag_citation"
    )
    assert draft.expected_behavior == record.expected_behavior
    assert any("Actual bad behavior to avoid" in item for item in draft.assertions)
    assert "regression" in draft.tags
    assert "from_bad_case" in draft.tags
    assert "rag_citation" in draft.tags


def test_mark_bad_case_regression_added_updates_status_and_case_id() -> None:
    record = _record("case_001", status="fixed")

    updated = mark_bad_case_regression_added(
        record,
        regression_case_id="agent_policy_refund_arrival_001_regression",
        regression_dataset_name="agent_eval",
    )

    assert updated.status == "regression_added"
    assert updated.regression_case_id == "agent_policy_refund_arrival_001_regression"
    assert updated.regression_dataset_name == "agent_eval"
    assert record.status == "fixed"


def _record(
    record_id: str,
    *,
    status: str = "open",
    severity: str = "critical",
    failure_layer: str = "rag_citation",
    regression_dataset_name: str | None = "agent_eval",
    source_case_id: str | None = "agent_policy_refund_arrival_001",
) -> BadCaseRecord:
    return BadCaseRecord(
        id=record_id,
        title="RAG source mismatch",
        source="eval",
        task_type="rag",
        severity=severity,
        status=status,
        dataset_name="agent_eval",
        dataset_version="stage6-v1",
        source_case_id=source_case_id,
        failure_layer=failure_layer,
        failure_category="rag_retrieval_or_citation",
        expected_behavior="answer with refund-return-policy.md",
        actual_behavior="answered with account-security-faq.md",
        recommended_action="Check retrieval and citation mapping.",
        regression_action="Add this case to regression dataset.",
        regression_dataset_name=regression_dataset_name,
        evidence_summary=(
            "expected_sources refund-return-policy.md but actual_sources "
            "account-security-faq.md"
        ),
        tags=["bad_case", "regression_candidate", failure_layer],
    )
