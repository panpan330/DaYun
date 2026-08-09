from pathlib import Path

from app.agents.eval_suite import AgentEvalRunReport, AgentEvalSuiteReport


def build_agent_eval_markdown_report(report: AgentEvalRunReport) -> str:
    lines = [
        "# Agent Evaluation Report",
        "",
        "## Overall",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Status", _status_label(report.passed)],
                ["Cases path", report.cases_path],
                ["Case filter", report.case_filter],
                ["Selected cases", str(report.selected_case_count)],
                ["Suites", ", ".join(suite.name for suite in report.suite_reports)],
                ["Suite count", str(report.suite_count)],
                ["Passed suites", str(report.passed_suite_count)],
                ["Failed suites", str(report.failed_suite_count)],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Suite Summary",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Suite", "Title", "Cases", "Failed cases", "Status"],
            [
                [
                    suite.name,
                    suite.title,
                    str(suite.case_count),
                    str(suite.failed_case_count),
                    _status_label(suite.passed),
                ]
                for suite in report.suite_reports
            ],
        )
    )

    for suite_report in report.suite_reports:
        lines.extend(["", *_suite_markdown_lines(suite_report)])

    return "\n".join(lines).rstrip() + "\n"


def write_agent_eval_markdown_report(
    report: AgentEvalRunReport,
    path: Path | str,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_agent_eval_markdown_report(report),
        encoding="utf-8",
    )
    return report_path


def _suite_markdown_lines(report: AgentEvalSuiteReport) -> list[str]:
    return [
        f"## {report.name}: {report.title}",
        "",
        "### Summary",
        "",
        "```text",
        *report.summary_lines,
        "```",
        "",
        "### Bad Cases",
        "",
        "```text",
        *report.bad_case_lines,
        "```",
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(_table_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(_table_cell(cell) for cell in row) + " |"
            for row in rows
        ],
    ]


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _status_label(passed: bool) -> str:
    return "PASS" if passed else "FAIL"
