from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from app.evaluation.production_regression import (
    ProductionRegressionOutcome,
    ProductionRegressionRun,
)


def build_production_regression_markdown_report(run: ProductionRegressionRun) -> str:
    lines = [
        "# Production Regression Report",
        "",
        "## Overall",
        "",
    ]
    passed_rate = (
        run.passed_case_count / run.total_case_count
        if run.total_case_count > 0
        else 0.0
    )
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Status", _status_label(run.passed)],
                ["Run id", run.run_id],
                ["Started at", run.started_at.isoformat()],
                ["Completed at", run.completed_at.isoformat()],
                ["Total cases", str(run.total_case_count)],
                ["Passed", str(run.passed_case_count)],
                ["Failed", str(run.failed_case_count)],
                ["Not ready", str(run.not_ready_case_count)],
                ["Error", str(run.error_case_count)],
                ["Passed %", f"{passed_rate * 100:.1f}%"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Assertion distribution",
            "",
            *_assertion_distribution_lines(run),
            "",
            "## Case details",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Bad case", "Title", "Assertion", "Expected", "Actual", "Outcome", "Detail"],
            [
                [
                    result.bad_case_id,
                    result.title,
                    result.assertion or "none",
                    result.expected or "",
                    result.actual or "",
                    result.outcome,
                    result.detail,
                ]
                for result in run.results
            ],
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def _assertion_distribution_lines(run: ProductionRegressionRun) -> list[str]:
    counts: DefaultDict[str, DefaultDict[ProductionRegressionOutcome, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for result in run.results:
        counts[result.assertion or "none"][result.outcome] += 1

    rows = [
        [
            assertion,
            str(counts[assertion]["passed"]),
            str(counts[assertion]["failed"]),
            str(counts[assertion]["not_ready"]),
            str(counts[assertion]["error"]),
        ]
        for assertion in sorted(counts)
    ]
    return _markdown_table(
        ["Assertion", "Passed", "Failed", "Not ready", "Error"],
        rows,
    )


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
