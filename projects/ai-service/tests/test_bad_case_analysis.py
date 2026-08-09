from pathlib import Path

from app.agents.bad_case_analysis import (
    analyze_agent_eval_bad_cases,
    build_bad_case_analysis_markdown_report,
    write_bad_case_analysis_markdown_report,
)
from app.agents.eval_suite import (
    AgentEvalRunReport,
    AgentEvalSuiteReport,
    run_agent_eval_suites,
)


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_analyze_agent_eval_bad_cases_handles_current_all_passed_report() -> None:
    run_report = run_agent_eval_suites(CASES_PATH)

    analysis = analyze_agent_eval_bad_cases(run_report)

    assert analysis.source_cases_path == str(CASES_PATH)
    assert analysis.failed_suite_count == 0
    assert analysis.bad_case_count == 0
    assert analysis.category_counts == {}
    assert analysis.items == []

    markdown = build_bad_case_analysis_markdown_report(analysis)
    assert markdown.startswith("# Agent Bad Case Analysis\n")
    assert "| Bad cases | 0 |" in markdown
    assert "No bad cases to analyze." in markdown


def test_analyze_agent_eval_bad_cases_classifies_rag_source_failure() -> None:
    run_report = _fake_failed_run_report(
        AgentEvalSuiteReport(
            name="rag",
            title="RAG + Agent evaluation",
            case_count=1,
            failed_case_count=1,
            passed=False,
            summary_lines=["RAG + Agent evaluation summary", "failed_cases: 1"],
            bad_case_lines=[
                "Bad cases:",
                "- agent_policy_refund_arrival_001: expected_status=answered actual_status=answered priority=p0",
                "  expected_sources: ['refund-return-policy.md']",
                "  actual_sources: ['account-security-faq.md']",
                "  - missing_sources=['refund-return-policy.md']",
            ],
        )
    )

    analysis = analyze_agent_eval_bad_cases(run_report)

    assert analysis.bad_case_count == 1
    assert analysis.category_counts == {"rag_retrieval_or_citation": 1}

    item = analysis.items[0]
    assert item.case_id == "agent_policy_refund_arrival_001"
    assert item.priority == "p0"
    assert item.category == "rag_retrieval_or_citation"
    assert item.likely_layer == "RAG retrieval or citation"
    assert "expected source" in item.diagnosis
    assert any("knowledge base" in question for question in item.review_questions)


def test_analyze_agent_eval_bad_cases_classifies_route_and_field_failures() -> None:
    run_report = _fake_failed_run_report(
        AgentEvalSuiteReport(
            name="route",
            title="Agent route evaluation",
            case_count=1,
            failed_case_count=1,
            passed=False,
            summary_lines=["Agent route evaluation summary", "failed_cases: 1"],
            bad_case_lines=[
                "Bad cases:",
                "- agent_policy_refund_arrival_001: priority=p0 task_type=policy_question terminal=extract_ticket_fields",
                "  expected_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need",
                "  actual_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need -> extract_ticket_fields",
                "  - visited_forbidden_nodes=['extract_ticket_fields']",
            ],
        ),
        AgentEvalSuiteReport(
            name="field",
            title="Ticket field evaluation",
            case_count=1,
            failed_case_count=1,
            passed=False,
            summary_lines=["Agent ticket field evaluation summary", "failed_cases: 1"],
            bad_case_lines=[
                "Bad cases:",
                "- agent_ticket_logistics_full_001: priority=p0 task_type=ticket_request field_accuracy=0.5000",
                "  - field order_id expected='ORDER1001' actual=None",
            ],
        ),
    )

    analysis = analyze_agent_eval_bad_cases(run_report)

    assert analysis.bad_case_count == 2
    assert analysis.category_counts == {
        "agent_routing": 1,
        "ticket_field_extraction": 1,
    }
    assert [item.category for item in analysis.items] == [
        "agent_routing",
        "ticket_field_extraction",
    ]


def test_build_bad_case_analysis_markdown_report_contains_evidence_and_actions() -> None:
    analysis = analyze_agent_eval_bad_cases(
        _fake_failed_run_report(
            AgentEvalSuiteReport(
                name="route",
                title="Agent route evaluation",
                case_count=1,
                failed_case_count=1,
                passed=False,
                summary_lines=["Agent route evaluation summary", "failed_cases: 1"],
                bad_case_lines=[
                    "Bad cases:",
                    "- agent_policy_refund_arrival_001: priority=p0 task_type=policy_question terminal=extract_ticket_fields",
                    "  expected_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need",
                    "  actual_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need -> extract_ticket_fields",
                    "  - visited_forbidden_nodes=['extract_ticket_fields']",
                ],
            )
        )
    )

    markdown = build_bad_case_analysis_markdown_report(
        analysis,
        title="Agent Bad Case Analysis Sample",
        note="This sample uses synthetic bad cases for learning.",
    )

    assert markdown.startswith("# Agent Bad Case Analysis Sample\n")
    assert "This sample uses synthetic bad cases for learning." in markdown
    assert "| agent_routing | 1 |" in markdown
    assert "### 1. route / agent_policy_refund_arrival_001" in markdown
    assert "#### Evidence" in markdown
    assert "visited_forbidden_nodes" in markdown
    assert "#### Review Questions" in markdown
    assert "#### Recommended Action" in markdown
    assert "#### Regression Action" in markdown


def test_write_bad_case_analysis_markdown_report_creates_parent_directory(
    tmp_path: Path,
) -> None:
    analysis = analyze_agent_eval_bad_cases(run_agent_eval_suites(CASES_PATH))
    report_path = tmp_path / "nested" / "bad_case_analysis.md"

    written_path = write_bad_case_analysis_markdown_report(analysis, report_path)

    assert written_path == report_path
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith(
        "# Agent Bad Case Analysis\n"
    )


def _fake_failed_run_report(
    *suite_reports: AgentEvalSuiteReport,
) -> AgentEvalRunReport:
    return AgentEvalRunReport(
        cases_path="memory",
        suite_count=len(suite_reports),
        passed_suite_count=0,
        failed_suite_count=len(suite_reports),
        passed=False,
        suite_reports=list(suite_reports),
    )
