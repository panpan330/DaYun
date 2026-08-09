from dataclasses import dataclass
from pathlib import Path

import pytest

from app.agents.eval_suite import (
    AgentEvalCaseFilter,
    AgentEvalSuite,
    describe_agent_eval_case_filter,
    exit_code_for_agent_eval_report,
    filter_agent_eval_cases,
    format_agent_eval_run_report,
    list_agent_eval_suite_names,
    resolve_agent_eval_suites,
    run_agent_eval_suites,
    run_agent_eval_suites_for_cases,
)
from app.agents.intent_evaluation import load_agent_eval_cases


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


@dataclass(frozen=True)
class FakeSummary:
    case_count: int
    failed_case_count: int


def test_list_agent_eval_suite_names_keeps_stable_learning_order() -> None:
    assert list_agent_eval_suite_names() == ["intent", "emotion", "field", "route", "rag", "multiturn"]


def test_resolve_agent_eval_suites_defaults_to_all_suites() -> None:
    suites = resolve_agent_eval_suites()

    assert [suite.name for suite in suites] == ["intent", "emotion", "field", "route", "rag", "multiturn"]


def test_run_agent_eval_suites_runs_all_current_suites() -> None:
    report = run_agent_eval_suites(CASES_PATH)

    assert report.cases_path == str(CASES_PATH)
    assert report.case_filter == "all"
    assert report.selected_case_count == 15
    assert report.suite_count == 6
    assert report.passed_suite_count == 6
    assert report.failed_suite_count == 0
    assert report.passed is True
    assert [suite_report.name for suite_report in report.suite_reports] == [
        "intent",
        "emotion",
        "field",
        "route",
        "rag",
        "multiturn",
    ]
    assert [suite_report.failed_case_count for suite_report in report.suite_reports] == [
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_run_agent_eval_suites_can_select_one_suite() -> None:
    report = run_agent_eval_suites(CASES_PATH, suite_names=["rag"])

    assert report.suite_count == 1
    assert report.passed_suite_count == 1
    assert report.suite_reports[0].name == "rag"
    assert report.suite_reports[0].case_count == 3
    assert report.suite_reports[0].passed is True


def test_run_agent_eval_suites_can_select_multiple_suites_in_requested_order() -> None:
    report = run_agent_eval_suites(CASES_PATH, suite_names=["route", "intent"])

    assert [suite_report.name for suite_report in report.suite_reports] == [
        "route",
        "intent",
    ]
    assert report.suite_count == 2
    assert report.failed_suite_count == 0


def test_filter_agent_eval_cases_selects_p0_regression_cases() -> None:
    cases = load_agent_eval_cases(CASES_PATH)
    case_filter = AgentEvalCaseFilter(
        include_tags=["regression"],
        priority="p0",
    )

    selected_cases = filter_agent_eval_cases(cases, case_filter=case_filter)

    assert len(selected_cases) == 13
    assert all(eval_case.metadata.priority == "p0" for eval_case in selected_cases)
    assert all("regression" in eval_case.metadata.tags for eval_case in selected_cases)
    assert "agent_smalltalk_hello_001" not in [
        eval_case.id for eval_case in selected_cases
    ]
    assert "agent_unclear_empty_001" not in [eval_case.id for eval_case in selected_cases]
    assert describe_agent_eval_case_filter(case_filter) == "tags=regression;priority=p0"


def test_run_agent_eval_suites_can_run_regression_case_filter() -> None:
    report = run_agent_eval_suites(
        CASES_PATH,
        suite_names=["intent", "rag"],
        case_filter=AgentEvalCaseFilter(
            include_tags=["regression"],
            priority="p0",
        ),
    )

    assert report.case_filter == "tags=regression;priority=p0"
    assert report.selected_case_count == 13
    assert [suite_report.name for suite_report in report.suite_reports] == [
        "intent",
        "rag",
    ]
    assert report.suite_reports[0].case_count == 13
    assert report.suite_reports[1].case_count == 3
    assert report.failed_suite_count == 0
    assert report.passed is True

    lines = format_agent_eval_run_report(report)
    assert "case_filter: tags=regression;priority=p0" in lines
    assert "selected_cases: 13" in lines


def test_run_agent_eval_suites_rejects_filter_that_selects_no_cases() -> None:
    with pytest.raises(ValueError, match="selected no cases"):
        run_agent_eval_suites(
            CASES_PATH,
            case_filter=AgentEvalCaseFilter(include_tags=["not_a_real_tag"]),
        )


def test_resolve_agent_eval_suites_rejects_unknown_suite_name() -> None:
    with pytest.raises(ValueError, match="Unknown eval suite"):
        resolve_agent_eval_suites(["unknown"])


def test_format_agent_eval_run_report_is_readable_for_humans_and_ci() -> None:
    report = run_agent_eval_suites(CASES_PATH, suite_names=["intent"])

    lines = format_agent_eval_run_report(report)

    assert lines[0] == "Agent evaluation suite"
    assert f"cases_path: {CASES_PATH}" in lines
    assert "suites: intent" in lines
    assert "== intent: Intent evaluation ==" in lines
    assert "Agent intent evaluation summary" in lines
    assert "No bad cases." in lines
    assert "Overall" in lines
    assert "failed_suites: 0" in lines
    assert "passed: true" in lines
    assert exit_code_for_agent_eval_report(report) == 0


def test_run_agent_eval_suites_marks_whole_run_failed_when_any_suite_fails() -> None:
    cases = load_agent_eval_cases(CASES_PATH)[:1]
    registry = {
        "fake": AgentEvalSuite(
            name="fake",
            title="Fake failing evaluation",
            evaluate=lambda _: FakeSummary(case_count=1, failed_case_count=1),
            format_summary=lambda _: ["Fake evaluation summary", "failed_cases: 1"],
            format_bad_cases=lambda _: ["Bad cases:", "- fake_case_001"],
        )
    }

    report = run_agent_eval_suites_for_cases(
        cases,
        cases_path="memory",
        suite_names=["fake"],
        registry=registry,
    )

    assert report.suite_count == 1
    assert report.passed_suite_count == 0
    assert report.failed_suite_count == 1
    assert report.passed is False
    assert report.suite_reports[0].passed is False
    assert exit_code_for_agent_eval_report(report) == 1
    assert "failed_suites: 1" in format_agent_eval_run_report(report)
