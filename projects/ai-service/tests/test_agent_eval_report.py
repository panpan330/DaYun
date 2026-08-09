from pathlib import Path

from app.agents.eval_report import (
    build_agent_eval_markdown_report,
    write_agent_eval_markdown_report,
)
from app.agents.eval_suite import (
    AgentEvalRunReport,
    AgentEvalSuiteReport,
    run_agent_eval_suites,
)


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_build_agent_eval_markdown_report_contains_overall_and_suite_tables() -> None:
    report = run_agent_eval_suites(CASES_PATH, suite_names=["intent", "rag"])

    markdown = build_agent_eval_markdown_report(report)

    assert markdown.startswith("# Agent Evaluation Report\n")
    assert "## Overall" in markdown
    assert "| Status | PASS |" in markdown
    assert f"| Cases path | {CASES_PATH} |" in markdown
    assert "| Case filter | all |" in markdown
    assert "| Selected cases | 15 |" in markdown
    assert "| Suites | intent, rag |" in markdown
    assert "## Suite Summary" in markdown
    assert "| intent | Intent evaluation | 15 | 0 | PASS |" in markdown
    assert "| rag | RAG + Agent evaluation | 3 | 0 | PASS |" in markdown
    assert "## intent: Intent evaluation" in markdown
    assert "## rag: RAG + Agent evaluation" in markdown
    assert "```text\nAgent intent evaluation summary" in markdown
    assert "```text\nNo bad cases.\n```" in markdown
    assert markdown.endswith("\n")


def test_build_agent_eval_markdown_report_marks_failed_run_and_suite() -> None:
    report = AgentEvalRunReport(
        cases_path="memory",
        suite_count=1,
        passed_suite_count=0,
        failed_suite_count=1,
        passed=False,
        suite_reports=[
            AgentEvalSuiteReport(
                name="fake",
                title="Fake failing evaluation",
                case_count=2,
                failed_case_count=1,
                passed=False,
                summary_lines=["Fake evaluation summary", "failed_cases: 1"],
                bad_case_lines=["Bad cases:", "- fake_case_001"],
            )
        ],
    )

    markdown = build_agent_eval_markdown_report(report)

    assert "| Status | FAIL |" in markdown
    assert "| fake | Fake failing evaluation | 2 | 1 | FAIL |" in markdown
    assert "Bad cases:\n- fake_case_001" in markdown


def test_write_agent_eval_markdown_report_creates_parent_directory(tmp_path: Path) -> None:
    report = run_agent_eval_suites(CASES_PATH, suite_names=["rag"])
    report_path = tmp_path / "nested" / "agent_eval_report.md"

    written_path = write_agent_eval_markdown_report(report, report_path)

    assert written_path == report_path
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith(
        "# Agent Evaluation Report\n"
    )
