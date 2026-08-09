from datetime import datetime, timezone

from app.evaluation.production_regression import (
    ProductionRegressionCaseResult,
    ProductionRegressionRun,
)
from app.evaluation.report_generator import build_production_regression_markdown_report


def _result(
    bad_case_id: str,
    *,
    title: str,
    outcome: str,
    assertion: str | None,
    expected: str | None = None,
    actual: str | None = None,
    detail: str = "Regression assertion passed.",
) -> ProductionRegressionCaseResult:
    return ProductionRegressionCaseResult(
        bad_case_id=bad_case_id,
        title=title,
        outcome=outcome,
        assertion=assertion,
        expected=expected,
        actual=actual,
        detail=detail,
    )


def _run() -> ProductionRegressionRun:
    return ProductionRegressionRun(
        run_id="reg-001",
        started_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 7, 12, 0, 5, tzinfo=timezone.utc),
        total_case_count=3,
        passed_case_count=2,
        failed_case_count=1,
        not_ready_case_count=0,
        error_case_count=0,
        passed=False,
        results=[
            _result(
                "bad-1",
                title="Refund requests must route to refund handling",
                outcome="passed",
                assertion="intent",
                expected="refund_request",
                actual="refund_request",
            ),
            _result(
                "bad-2",
                title="Refund tool must be called",
                outcome="passed",
                assertion="tool_called",
                expected="refund_order",
                actual="execute_refund_request",
            ),
            _result(
                "bad-3",
                title="Refund requests must not be misclassified",
                outcome="failed",
                assertion="intent",
                expected="refund_request",
                actual="order_query",
                detail="Regression assertion failed.",
            ),
        ],
    )


def test_build_production_regression_markdown_report_includes_overview_and_assertions() -> None:
    markdown = build_production_regression_markdown_report(_run())

    assert markdown.startswith("# Production Regression Report\n")
    assert "## Overall" in markdown
    assert "| Status | FAIL |" in markdown
    assert "| Run id | reg-001 |" in markdown
    assert "| Total cases | 3 |" in markdown
    assert "| Passed | 2 |" in markdown
    assert "| Failed | 1 |" in markdown
    assert "| Passed % | 66.7% |" in markdown

    assert "## Assertion distribution" in markdown
    assert "| intent | 1 | 1 | 0 | 0 |" in markdown
    assert "| tool_called | 1 | 0 | 0 | 0 |" in markdown

    assert "## Case details" in markdown
    assert "| bad-1 | Refund requests must route to refund handling | intent | refund_request | refund_request | passed |" in markdown
    assert "| bad-2 | Refund tool must be called | tool_called | refund_order | execute_refund_request | passed |" in markdown
    assert "| bad-3 | Refund requests must not be misclassified | intent | refund_request | order_query | failed |" in markdown


def test_build_production_regression_markdown_report_handles_empty_results() -> None:
    run = ProductionRegressionRun(
        run_id="reg-empty",
        started_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 8, 7, 12, 0, 5, tzinfo=timezone.utc),
        total_case_count=0,
        passed_case_count=0,
        failed_case_count=0,
        not_ready_case_count=0,
        error_case_count=0,
        passed=True,
        results=[],
    )

    markdown = build_production_regression_markdown_report(run)

    assert "| Status | PASS |" in markdown
    assert "| Passed % | 0.0% |" in markdown
    assert "## Assertion distribution" in markdown
    assert "## Case details" in markdown
