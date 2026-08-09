import json
from pathlib import Path

import pytest

from app.agents.intent_evaluation import (
    AgentEvalDataset,
    evaluate_intent_case,
    evaluate_intent_cases,
    format_intent_bad_cases,
    format_intent_eval_summary,
    load_agent_eval_cases,
    load_agent_eval_dataset,
)


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_load_agent_eval_dataset_validates_first_version_cases() -> None:
    dataset = load_agent_eval_dataset(CASES_PATH)

    assert dataset.schema_version == "stage6.agent_eval.v1"
    assert len(dataset.cases) == 15
    assert dataset.cases[0].id == "agent_policy_refund_arrival_001"
    assert dataset.cases[0].expected.intent == "policy_question"
    assert dataset.cases[0].expected.intent_route == "retrieve_policy"


def test_evaluate_intent_case_marks_pass_when_classifier_matches_expected() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[3]

    result = evaluate_intent_case(eval_case)

    assert result.case_id == "agent_order_query_with_order_id_001"
    assert result.expected_intent == "order_query"
    assert result.actual_intent == "order_query"
    assert result.expected_route == "query_order"
    assert result.actual_route == "query_order"
    assert result.passed is True
    assert result.failed_reason is None


def test_evaluate_intent_case_marks_bad_case_when_classifier_is_wrong() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[0]

    result = evaluate_intent_case(
        eval_case,
        classifier=lambda _: {
            "intent": "unclear",
            "reason": "demo classifier returned the wrong intent",
        },
    )

    assert result.passed is False
    assert result.expected_intent == "policy_question"
    assert result.actual_intent == "unclear"
    assert result.expected_route == "retrieve_policy"
    assert result.actual_route == "ask_clarifying_question"
    assert "expected intent=policy_question" in result.failed_reason


def test_evaluate_intent_cases_summarizes_accuracy_and_p0_accuracy() -> None:
    cases = load_agent_eval_cases(CASES_PATH)

    summary = evaluate_intent_cases(cases)

    assert summary.case_count == 15
    assert summary.passed_case_count == 15
    assert summary.failed_case_count == 0
    assert summary.accuracy == 1.0
    assert summary.p0_case_count == 13
    assert summary.p0_failed_case_count == 0
    assert summary.p0_accuracy == 1.0


def test_format_intent_eval_summary_and_bad_cases_are_readable() -> None:
    cases = load_agent_eval_cases(CASES_PATH)[:2]

    summary = evaluate_intent_cases(
        cases,
        classifier=lambda _: {
            "intent": "unclear",
            "reason": "demo classifier returned the wrong intent",
        },
    )

    summary_lines = format_intent_eval_summary(summary)
    bad_case_lines = format_intent_bad_cases(summary)

    assert "accuracy: 0.0000" in summary_lines
    assert "p0_accuracy: 0.0000" in summary_lines
    assert bad_case_lines[0] == "Bad cases:"
    assert "agent_policy_refund_arrival_001" in bad_case_lines[1]


def test_format_intent_bad_cases_handles_all_passed_summary() -> None:
    summary = evaluate_intent_cases(load_agent_eval_cases(CASES_PATH))

    assert format_intent_bad_cases(summary) == ["No bad cases."]


def test_agent_eval_dataset_rejects_duplicate_case_ids_and_wrong_routes(
    tmp_path: Path,
) -> None:
    raw_dataset = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    duplicate_dataset = dict(raw_dataset)
    duplicate_dataset["cases"] = [raw_dataset["cases"][0], raw_dataset["cases"][0]]
    duplicate_path = tmp_path / "duplicate_cases.json"
    duplicate_path.write_text(
        json.dumps(duplicate_dataset, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_agent_eval_dataset(duplicate_path)

    wrong_route_dataset = dict(raw_dataset)
    wrong_route_case = dict(raw_dataset["cases"][0])
    wrong_route_case["expected"] = dict(wrong_route_case["expected"])
    wrong_route_case["expected"]["intent_route"] = "query_order"
    wrong_route_dataset["cases"] = [wrong_route_case]

    with pytest.raises(ValueError, match="intent_route"):
        AgentEvalDataset.model_validate(wrong_route_dataset)
