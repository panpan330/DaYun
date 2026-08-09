from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.bad_case_analysis import (  # noqa: E402
    analyze_agent_eval_bad_cases,
    write_bad_case_analysis_markdown_report,
)
from app.agents.eval_suite import (  # noqa: E402
    AgentEvalCaseFilter,
    exit_code_for_agent_eval_report,
    format_agent_eval_run_report,
    list_agent_eval_suite_names,
    run_agent_eval_suites,
)
from app.agents.eval_report import write_agent_eval_markdown_report  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "agent_eval" / "agent_cases.json"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run local Agent evaluation suites against a fixed JSON dataset."
    )
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the Agent evaluation JSON dataset.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["all", *list_agent_eval_suite_names()],
        help="Evaluation suite to run. Repeat this option to run multiple suites.",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print available suite names and exit.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Write a Markdown evaluation report to this path.",
    )
    parser.add_argument(
        "--bad-case-analysis-path",
        type=Path,
        help="Write a Markdown bad case analysis report to this path.",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Run only cases tagged as regression cases.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        help="Run only cases containing this tag. Repeat to require multiple tags.",
    )
    parser.add_argument(
        "--priority",
        choices=["p0", "p1", "p2"],
        help="Run only cases with this priority.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_suites:
        for suite_name in list_agent_eval_suite_names():
            print(suite_name)
        return 0

    report = run_agent_eval_suites(
        args.cases_path,
        suite_names=args.suite,
        case_filter=_build_case_filter(
            regression=args.regression,
            tags=args.tag,
            priority=args.priority,
        ),
    )
    for line in format_agent_eval_run_report(report):
        print(line)
    if args.report_path is not None:
        report_path = write_agent_eval_markdown_report(report, args.report_path)
        print("")
        print(f"markdown_report: {report_path}")
    if args.bad_case_analysis_path is not None:
        analysis = analyze_agent_eval_bad_cases(report)
        analysis_path = write_bad_case_analysis_markdown_report(
            analysis,
            args.bad_case_analysis_path,
        )
        print("")
        print(f"bad_case_analysis: {analysis_path}")
    return exit_code_for_agent_eval_report(report)


def _build_case_filter(
    *,
    regression: bool,
    tags: Sequence[str] | None,
    priority: str | None,
) -> AgentEvalCaseFilter | None:
    include_tags = list(tags or [])
    if regression and "regression" not in include_tags:
        include_tags.append("regression")
    if not include_tags and priority is None:
        return None
    return AgentEvalCaseFilter(
        include_tags=include_tags,
        priority=priority,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    raise SystemExit(main())
