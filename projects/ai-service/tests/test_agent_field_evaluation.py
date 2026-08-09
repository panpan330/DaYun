from pathlib import Path

import pytest

from app.agents.field_evaluation import (
    evaluate_ticket_field_case,
    evaluate_ticket_field_cases,
    format_ticket_field_bad_cases,
    format_ticket_field_eval_summary,
    select_ticket_field_eval_cases,
)
from app.agents.intent_evaluation import load_agent_eval_cases


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_select_ticket_field_eval_cases_only_keeps_ticket_expected_cases() -> None:
    cases = load_agent_eval_cases(CASES_PATH)

    ticket_cases = select_ticket_field_eval_cases(cases)

    assert [eval_case.id for eval_case in ticket_cases] == [
        "agent_no_context_membership_points_001",
        "agent_ticket_logistics_full_001",
        "agent_ticket_complaint_missing_order_001",
        "agent_ticket_refund_full_001",
    ]


def test_evaluate_ticket_field_case_handles_no_context_policy_gap() -> None:
    eval_case = select_ticket_field_eval_cases(load_agent_eval_cases(CASES_PATH))[0]

    result = evaluate_ticket_field_case(eval_case)

    assert result.case_id == "agent_no_context_membership_points_001"
    assert result.actual_should_create_ticket is True
    assert result.actual_ticket_need_source == "rag_no_context"
    assert result.expected_fields == {"issue_type": "policy_gap"}
    assert result.actual_fields["issue_type"] == "policy_gap"
    assert result.actual_missing_ticket_fields == []
    assert result.actual_confirmation_required is True
    assert result.field_accuracy == 1.0
    assert result.passed is True


def test_evaluate_ticket_field_case_matches_complete_logistics_fields() -> None:
    eval_case = select_ticket_field_eval_cases(load_agent_eval_cases(CASES_PATH))[1]

    result = evaluate_ticket_field_case(eval_case)

    assert result.case_id == "agent_ticket_logistics_full_001"
    assert result.expected_fields == {
        "issue_type": "logistics",
        "order_id": "A1002",
        "user_request": "创建工单",
        "urgency": "high",
        "need_human_review": True,
    }
    assert result.actual_fields["issue_type"] == "logistics"
    assert result.actual_fields["order_id"] == "A1002"
    assert result.actual_fields["user_request"] == "创建工单"
    assert result.actual_fields["urgency"] == "high"
    assert result.actual_fields["need_human_review"] is True
    assert result.actual_missing_ticket_fields == []
    assert result.actual_confirmation_required is True
    assert result.passed is True


def test_evaluate_ticket_field_case_matches_missing_order_id_case() -> None:
    eval_case = select_ticket_field_eval_cases(load_agent_eval_cases(CASES_PATH))[2]

    result = evaluate_ticket_field_case(eval_case)

    assert result.case_id == "agent_ticket_complaint_missing_order_001"
    assert result.actual_fields["issue_type"] == "complaint"
    assert result.actual_fields["order_id"] is None
    assert result.actual_fields["user_request"] == "投诉处理"
    assert result.actual_fields["urgency"] == "normal"
    assert result.actual_missing_ticket_fields == ["order_id"]
    assert result.actual_confirmation_required is False
    assert result.passed is True


def test_evaluate_ticket_field_case_marks_bad_case_when_field_is_wrong() -> None:
    eval_case = select_ticket_field_eval_cases(load_agent_eval_cases(CASES_PATH))[1]

    result = evaluate_ticket_field_case(
        eval_case,
        agent_runner=lambda _: {
            "needs_ticket": True,
            "ticket_need_source": "explicit_user_request",
            "ticket_fields": {
                "issue_type": "refund",
                "order_id": "A1002",
                "user_request": "创建工单",
                "urgency": "normal",
                "need_human_review": True,
            },
            "missing_ticket_fields": [],
            "ticket_confirmation_required": True,
        },
    )

    assert result.passed is False
    assert result.matched_field_count == 3
    assert result.field_accuracy == 0.6
    assert any("issue_type" in reason for reason in result.failed_reasons)
    assert any("urgency" in reason for reason in result.failed_reasons)


def test_evaluate_ticket_field_cases_summarizes_case_and_field_accuracy() -> None:
    summary = evaluate_ticket_field_cases(load_agent_eval_cases(CASES_PATH))

    assert summary.case_count == 4
    assert summary.passed_case_count == 4
    assert summary.failed_case_count == 0
    assert summary.case_pass_rate == 1.0
    assert summary.expected_field_count == 16
    assert summary.matched_field_count == 16
    assert summary.field_accuracy == 1.0
    assert summary.p0_case_count == 4
    assert summary.p0_failed_case_count == 0
    assert summary.p0_case_pass_rate == 1.0
    assert summary.missing_field_case_count == 1
    assert summary.missing_field_passed_case_count == 1


def test_format_ticket_field_eval_summary_and_bad_cases_are_readable() -> None:
    cases = select_ticket_field_eval_cases(load_agent_eval_cases(CASES_PATH))[:1]
    summary = evaluate_ticket_field_cases(
        cases,
        agent_runner=lambda _: {
            "needs_ticket": False,
            "ticket_fields": {},
            "missing_ticket_fields": [],
        },
    )

    summary_lines = format_ticket_field_eval_summary(summary)
    bad_case_lines = format_ticket_field_bad_cases(summary)

    assert "case_pass_rate: 0.0000" in summary_lines
    assert "field_accuracy: 0.0000" in summary_lines
    assert bad_case_lines[0] == "Bad cases:"
    assert "agent_no_context_membership_points_001" in bad_case_lines[1]
    assert any("should_create_ticket" in line for line in bad_case_lines)


def test_format_ticket_field_bad_cases_handles_all_passed_summary() -> None:
    summary = evaluate_ticket_field_cases(load_agent_eval_cases(CASES_PATH))

    assert format_ticket_field_bad_cases(summary) == ["No bad cases."]


def test_evaluate_ticket_field_case_rejects_non_ticket_case() -> None:
    non_ticket_case = load_agent_eval_cases(CASES_PATH)[0]

    with pytest.raises(ValueError, match="ticket creation"):
        evaluate_ticket_field_case(non_ticket_case)
