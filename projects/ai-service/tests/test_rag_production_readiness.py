import pytest

from app.rag.production_readiness import (
    RagProductionReadinessAnswer,
    RagProductionReadinessCheck,
    build_rag_production_readiness_report,
    default_rag_production_readiness_checklist,
    format_rag_production_readiness_report,
)


def _passed_answers() -> list[RagProductionReadinessAnswer]:
    return [
        RagProductionReadinessAnswer(
            check_id=check.check_id,
            status="passed",
            evidence=[f"{check.check_id} evidence"],
        )
        for check in default_rag_production_readiness_checklist()
    ]


def test_default_rag_production_readiness_checklist_covers_core_categories() -> None:
    checklist = default_rag_production_readiness_checklist()

    categories = {check.category for check in checklist}
    blocker_ids = {check.check_id for check in checklist if check.release_blocker}

    assert categories == {
        "quality",
        "security",
        "performance",
        "cost",
        "observability",
        "data",
        "agent_boundary",
    }
    assert "quality.retrieval_metrics" in blocker_ids
    assert "security.permission_filter" in blocker_ids
    assert "performance.degradation" in blocker_ids
    assert "data.update_plan" in blocker_ids
    assert "agent_boundary.write_confirmation" in blocker_ids


def test_build_rag_production_readiness_report_is_ready_when_all_checks_pass() -> None:
    report = build_rag_production_readiness_report(_passed_answers())
    lines = format_rag_production_readiness_report(report)

    assert report.release_status == "ready"
    assert report.passed_count == report.checklist_count
    assert report.blocker_count == 0
    assert report.findings == []
    assert "release_status: ready" in lines


def test_build_rag_production_readiness_report_blocks_when_required_check_missing() -> None:
    answers = [
        answer
        for answer in _passed_answers()
        if answer.check_id != "security.permission_filter"
    ]

    report = build_rag_production_readiness_report(answers)

    assert report.release_status == "blocked"
    assert report.not_checked_count == 1
    assert report.blocker_check_ids == ["security.permission_filter"]
    assert report.findings[0].recommendation == "Collect evidence before release."


def test_build_rag_production_readiness_report_blocks_when_required_check_failed() -> None:
    answers = _passed_answers()
    failed_index = next(
        index
        for index, answer in enumerate(answers)
        if answer.check_id == "quality.answer_quality"
    )
    answers[failed_index] = RagProductionReadinessAnswer(
        check_id="quality.answer_quality",
        status="failed",
        evidence=["answer quality failed on refund cases"],
    )

    report = build_rag_production_readiness_report(answers)

    assert report.release_status == "blocked"
    assert report.failed_count == 1
    assert report.blocker_check_ids == ["quality.answer_quality"]
    assert report.findings[0].recommendation == "Fix this blocker before release."


def test_build_rag_production_readiness_report_is_conditional_for_non_blocker_warning() -> None:
    answers = _passed_answers()
    warning_index = next(
        index
        for index, answer in enumerate(answers)
        if answer.check_id == "cost.provider_fallback"
    )
    answers[warning_index] = RagProductionReadinessAnswer(
        check_id="cost.provider_fallback",
        status="warning",
        evidence=["rerank fallback rate is higher than expected"],
    )

    report = build_rag_production_readiness_report(answers)

    assert report.release_status == "conditional"
    assert report.warning_count == 1
    assert report.blocker_count == 0
    assert report.findings[0].check_id == "cost.provider_fallback"


def test_build_rag_production_readiness_report_rejects_unknown_or_duplicate_answers() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_rag_production_readiness_report(
            [
                RagProductionReadinessAnswer(
                    check_id="unknown.check",
                    status="passed",
                )
            ]
        )

    with pytest.raises(ValueError, match="duplicated"):
        build_rag_production_readiness_report(
            [
                RagProductionReadinessAnswer(
                    check_id="quality.retrieval_metrics",
                    status="passed",
                ),
                RagProductionReadinessAnswer(
                    check_id="quality.retrieval_metrics",
                    status="warning",
                ),
            ]
        )


def test_build_rag_production_readiness_report_rejects_duplicate_check_ids() -> None:
    duplicated_check = RagProductionReadinessCheck(
        check_id="same",
        category="quality",
        title="First",
        requirement="First requirement",
        risk_if_missing="First risk",
    )

    with pytest.raises(ValueError, match="duplicated"):
        build_rag_production_readiness_report(
            [],
            checklist=[
                duplicated_check,
                duplicated_check.model_copy(update={"title": "Second"}),
            ],
        )
