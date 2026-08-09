"""Run production bad-case regression and write a Markdown report.

Usage:
    uv run python scripts/production_regression.py \
        --report-path data/agent_eval/reports/production_regression_report.md

Reads the checked-in bad case registry, runs the supervisor-defined production
regression assertions for every record with source="production" and
status="regression_added", appends the run to the run history, and writes a
Markdown report. An empty or missing registry simply produces a zero-case run.
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.bad_case_registry import (  # noqa: E402
    BadCaseRegistry,
)
from app.evaluation.production_regression import (  # noqa: E402
    ProductionRegressionRun,
    run_production_bad_case_regression,
)
from app.evaluation.production_regression_history import (  # noqa: E402
    append_production_regression_run,
)
from app.evaluation.report_generator import (  # noqa: E402
    build_production_regression_markdown_report,
)


DEFAULT_BAD_CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "bad_cases.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "evaluation" / "production_regression_runs.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data" / "agent_eval" / "reports" / "production_regression_report.md"
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run production bad-case regression assertions and write a Markdown report."
    )
    parser.add_argument(
        "--bad-cases-path",
        type=Path,
        default=DEFAULT_BAD_CASES_PATH,
        help="Path to the bad case registry JSON file.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to the production regression run history JSON file.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to write the Markdown regression report to.",
    )
    return parser


def load_bad_case_registry(path: Path) -> BadCaseRegistry:
    """Load the bad case registry, tolerating a missing or empty file."""
    if not path.exists():
        return BadCaseRegistry(schema_version="stage10.bad_case_registry.v1")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return BadCaseRegistry(schema_version="stage10.bad_case_registry.v1")
    return BadCaseRegistry.model_validate_json(text)


def main(argv: list[str] | None = None) -> int:
    args: Namespace = build_parser().parse_args(argv)
    bad_cases_path: Path = args.bad_cases_path
    history_path: Path = args.history_path
    report_path: Path = args.report_path

    registry = load_bad_case_registry(bad_cases_path)
    run: ProductionRegressionRun = run_production_bad_case_regression(registry.records)

    append_production_regression_run(history_path, run)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_production_regression_markdown_report(run),
        encoding="utf-8",
    )

    print(f"production_regression: {run.run_id}")
    print(
        f"passed={run.passed_case_count} failed={run.failed_case_count} "
        f"not_ready={run.not_ready_case_count} error={run.error_case_count} "
        f"total={run.total_case_count}"
    )
    print(f"report: {report_path}")
    # 空 run（registry 中无 production regression_added 记录）视为通过，返回 0；
    # 非空但 failed/not_ready/error 时返回 1，让 CI 的 eval-regression job 变红。
    exit_code = 0 if (run.passed or run.total_case_count == 0) else 1
    print(f"exit_code: {exit_code} (0 = pass, 1 = regression failed)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
