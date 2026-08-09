from pathlib import Path

import pytest

from app.agents.intent_evaluation import load_agent_eval_cases
from app.agents.rag_agent_evaluation import (
    evaluate_rag_agent_case,
    evaluate_rag_agent_cases,
    format_rag_agent_bad_cases,
    format_rag_agent_eval_summary,
    select_rag_agent_eval_cases,
)


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_select_rag_agent_eval_cases_only_keeps_rag_expected_cases() -> None:
    cases = load_agent_eval_cases(CASES_PATH)

    rag_cases = select_rag_agent_eval_cases(cases)

    assert [eval_case.id for eval_case in rag_cases] == [
        "agent_policy_refund_arrival_001",
        "agent_policy_account_security_001",
        "agent_no_context_membership_points_001",
    ]


def test_evaluate_rag_agent_case_matches_refund_answered_behavior() -> None:
    eval_case = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[0]

    result = evaluate_rag_agent_case(eval_case)

    assert result.case_id == "agent_policy_refund_arrival_001"
    assert result.expected_rag_answer_status == "answered"
    assert result.actual_rag_answer_status == "answered"
    assert result.expected_sources == ["refund-return-policy.md"]
    assert result.actual_sources == ["refund-return-policy.md"]
    assert result.matched_sources == ["refund-return-policy.md"]
    assert result.citations_present is True
    assert result.actual_should_create_ticket is False
    assert result.actual_ticket_need_source == "rag_answered"
    assert result.passed is True


def test_evaluate_rag_agent_case_matches_account_security_answered_behavior() -> None:
    eval_case = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[1]

    result = evaluate_rag_agent_case(eval_case)

    assert result.case_id == "agent_policy_account_security_001"
    assert result.actual_rag_answer_status == "answered"
    assert result.actual_sources == ["account-security-faq.md"]
    assert result.actual_should_create_ticket is False
    assert result.actual_ticket_need_source == "rag_answered"
    assert result.node_history == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "retrieve_policy",
        "decide_ticket_need",
    ]
    assert result.passed is True


def test_evaluate_rag_agent_case_matches_no_context_to_policy_gap_ticket() -> None:
    eval_case = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[2]

    result = evaluate_rag_agent_case(eval_case)

    assert result.case_id == "agent_no_context_membership_points_001"
    assert result.expected_rag_answer_status == "no_context"
    assert result.actual_rag_answer_status == "no_context"
    assert result.expected_sources == []
    assert result.actual_sources == []
    assert result.actual_should_create_ticket is True
    assert result.actual_ticket_need_source == "rag_no_context"
    assert result.actual_issue_type == "policy_gap"
    assert result.actual_confirmation_required is True
    assert result.no_context_behavior_passed is True
    assert result.passed is True


def test_evaluate_rag_agent_case_marks_bad_case_for_wrong_source() -> None:
    eval_case = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[0]

    result = evaluate_rag_agent_case(
        eval_case,
        agent_runner=lambda _: {
            "rag_answer_status": "answered",
            "rag_citations": [{"source": "account-security-faq.md"}],
            "needs_ticket": False,
            "ticket_need_source": "rag_answered",
            "node_history": ["normalize_user_input", "emotion_recognition", "classify_intent"],
        },
    )

    assert result.passed is False
    assert result.missing_sources == ["refund-return-policy.md"]
    assert "account-security-faq.md" in result.unexpected_sources
    assert any("missing_sources" in reason for reason in result.failed_reasons)


def test_evaluate_rag_agent_case_marks_bad_case_for_no_context_without_ticket() -> None:
    eval_case = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[2]

    result = evaluate_rag_agent_case(
        eval_case,
        agent_runner=lambda _: {
            "rag_answer_status": "no_context",
            "rag_citations": [],
            "needs_ticket": False,
            "ticket_need_source": "rag_answered",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "retrieve_policy",
                "decide_ticket_need",
            ],
        },
    )

    assert result.passed is False
    assert result.ticket_decision_passed is False
    assert result.no_context_behavior_passed is False
    assert any("should_create_ticket" in reason for reason in result.failed_reasons)
    assert any("rag_no_context" in reason for reason in result.failed_reasons)


def test_evaluate_rag_agent_cases_summarizes_current_rag_cases() -> None:
    summary = evaluate_rag_agent_cases(load_agent_eval_cases(CASES_PATH))

    assert summary.case_count == 3
    assert summary.passed_case_count == 3
    assert summary.failed_case_count == 0
    assert summary.case_pass_rate == 1.0
    assert summary.answered_case_count == 2
    assert summary.answered_passed_case_count == 2
    assert summary.no_context_case_count == 1
    assert summary.no_context_passed_case_count == 1
    assert summary.expected_source_count == 2
    assert summary.matched_source_count == 2
    assert summary.source_recall == 1.0
    assert summary.citation_passed_count == 3
    assert summary.ticket_decision_passed_count == 3
    assert summary.p0_case_count == 3
    assert summary.p0_failed_case_count == 0
    assert summary.p0_case_pass_rate == 1.0


def test_format_rag_agent_eval_summary_and_bad_cases_are_readable() -> None:
    cases = select_rag_agent_eval_cases(load_agent_eval_cases(CASES_PATH))[:1]
    summary = evaluate_rag_agent_cases(
        cases,
        agent_runner=lambda _: {
            "rag_answer_status": "no_context",
            "rag_citations": [],
            "needs_ticket": True,
            "ticket_need_source": "rag_no_context",
            "ticket_fields": {"issue_type": "policy_gap"},
            "node_history": [],
        },
    )

    summary_lines = format_rag_agent_eval_summary(summary)
    bad_case_lines = format_rag_agent_bad_cases(summary)

    assert "case_pass_rate: 0.0000" in summary_lines
    assert "source_recall: 0.0000" in summary_lines
    assert bad_case_lines[0] == "Bad cases:"
    assert "agent_policy_refund_arrival_001" in bad_case_lines[1]
    assert any("expected_sources:" in line for line in bad_case_lines)
    assert any("actual_sources:" in line for line in bad_case_lines)


def test_format_rag_agent_bad_cases_handles_all_passed_summary() -> None:
    summary = evaluate_rag_agent_cases(load_agent_eval_cases(CASES_PATH))

    assert format_rag_agent_bad_cases(summary) == ["No bad cases."]


def test_evaluate_rag_agent_case_rejects_non_rag_case() -> None:
    non_rag_case = load_agent_eval_cases(CASES_PATH)[3]

    with pytest.raises(ValueError, match="expected.rag"):
        evaluate_rag_agent_case(non_rag_case)
