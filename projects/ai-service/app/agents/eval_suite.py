from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.agents.field_evaluation import (
    evaluate_ticket_field_cases,
    format_ticket_field_bad_cases,
    format_ticket_field_eval_summary,
)
from app.agents.emotion_evaluation import (
    evaluate_emotion_cases,
    format_emotion_bad_cases,
    format_emotion_eval_summary,
)
from app.agents.intent_evaluation import (
    AgentEvalCase,
    AgentEvalPriority,
    evaluate_intent_cases,
    format_intent_bad_cases,
    format_intent_eval_summary,
    load_agent_eval_cases,
)
from app.agents.multiturn_evaluation import (
    evaluate_multiturn_cases,
    format_multiturn_bad_cases,
    format_multiturn_eval_summary,
)
from app.agents.rag_agent_evaluation import (
    evaluate_rag_agent_cases,
    format_rag_agent_bad_cases,
    format_rag_agent_eval_summary,
)
from app.agents.route_evaluation import (
    evaluate_agent_route_cases,
    format_agent_route_bad_cases,
    format_agent_route_eval_summary,
)
from app.schemas.multiturn_eval import load_multiturn_cases


DEFAULT_MULTITURN_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "agent_eval" / "multiturn_cases.json"


EvaluationSummary = Any
EvaluateSuite = Callable[[Sequence[AgentEvalCase]], EvaluationSummary]
FormatSuiteReport = Callable[[EvaluationSummary], list[str]]


@dataclass(frozen=True)
class AgentEvalSuite:
    name: str
    title: str
    evaluate: EvaluateSuite
    format_summary: FormatSuiteReport
    format_bad_cases: FormatSuiteReport


class AgentEvalSuiteReport(BaseModel):
    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    passed: bool
    summary_lines: list[str] = Field(default_factory=list)
    bad_case_lines: list[str] = Field(default_factory=list)


class AgentEvalCaseFilter(BaseModel):
    include_tags: list[str] = Field(default_factory=list)
    priority: AgentEvalPriority | None = None


class AgentEvalRunReport(BaseModel):
    cases_path: str = Field(min_length=1)
    selected_case_count: int = Field(default=0, ge=0)
    case_filter: str = "all"
    suite_count: int = Field(ge=0)
    passed_suite_count: int = Field(ge=0)
    failed_suite_count: int = Field(ge=0)
    passed: bool
    suite_reports: list[AgentEvalSuiteReport] = Field(default_factory=list)


def _evaluate_multiturn_suite(cases: Sequence[AgentEvalCase]) -> Any:
    """Multiturn suite loads its own dataset; the passed single-turn cases are ignored."""
    del cases
    multiturn_cases = load_multiturn_cases(DEFAULT_MULTITURN_CASES_PATH)
    return evaluate_multiturn_cases(multiturn_cases)


def build_agent_eval_suite_registry() -> dict[str, AgentEvalSuite]:
    suites = [
        AgentEvalSuite(
            name="intent",
            title="Intent evaluation",
            evaluate=evaluate_intent_cases,
            format_summary=format_intent_eval_summary,
            format_bad_cases=format_intent_bad_cases,
        ),
        AgentEvalSuite(
            name="emotion",
            title="Emotion recognition evaluation",
            evaluate=evaluate_emotion_cases,
            format_summary=format_emotion_eval_summary,
            format_bad_cases=format_emotion_bad_cases,
        ),
        AgentEvalSuite(
            name="field",
            title="Ticket field evaluation",
            evaluate=evaluate_ticket_field_cases,
            format_summary=format_ticket_field_eval_summary,
            format_bad_cases=format_ticket_field_bad_cases,
        ),
        AgentEvalSuite(
            name="route",
            title="Agent route evaluation",
            evaluate=evaluate_agent_route_cases,
            format_summary=format_agent_route_eval_summary,
            format_bad_cases=format_agent_route_bad_cases,
        ),
        AgentEvalSuite(
            name="rag",
            title="RAG + Agent evaluation",
            evaluate=evaluate_rag_agent_cases,
            format_summary=format_rag_agent_eval_summary,
            format_bad_cases=format_rag_agent_bad_cases,
        ),
        AgentEvalSuite(
            name="multiturn",
            title="Multiturn conversation evaluation",
            evaluate=_evaluate_multiturn_suite,
            format_summary=format_multiturn_eval_summary,
            format_bad_cases=format_multiturn_bad_cases,
        ),
    ]
    return {suite.name: suite for suite in suites}


def list_agent_eval_suite_names(
    registry: Mapping[str, AgentEvalSuite] | None = None,
) -> list[str]:
    return list((registry or build_agent_eval_suite_registry()).keys())


def resolve_agent_eval_suites(
    suite_names: Sequence[str] | None = None,
    *,
    registry: Mapping[str, AgentEvalSuite] | None = None,
) -> list[AgentEvalSuite]:
    suite_registry = registry or build_agent_eval_suite_registry()
    if not suite_names or "all" in suite_names:
        return list(suite_registry.values())

    unknown_names = [name for name in suite_names if name not in suite_registry]
    if unknown_names:
        available = ", ".join(["all", *suite_registry.keys()])
        unknown = ", ".join(unknown_names)
        raise ValueError(f"Unknown eval suite: {unknown}. Available suites: {available}")

    return [suite_registry[name] for name in suite_names]


def run_agent_eval_suites(
    cases_path: Path | str,
    *,
    suite_names: Sequence[str] | None = None,
    case_filter: AgentEvalCaseFilter | None = None,
    registry: Mapping[str, AgentEvalSuite] | None = None,
) -> AgentEvalRunReport:
    cases = load_agent_eval_cases(cases_path)
    return run_agent_eval_suites_for_cases(
        cases,
        cases_path=str(cases_path),
        suite_names=suite_names,
        case_filter=case_filter,
        registry=registry,
    )


def run_agent_eval_suites_for_cases(
    cases: Sequence[AgentEvalCase],
    *,
    cases_path: str,
    suite_names: Sequence[str] | None = None,
    case_filter: AgentEvalCaseFilter | None = None,
    registry: Mapping[str, AgentEvalSuite] | None = None,
) -> AgentEvalRunReport:
    selected_cases = filter_agent_eval_cases(cases, case_filter=case_filter)
    if not selected_cases:
        raise ValueError(
            f"agent eval case filter selected no cases: "
            f"{describe_agent_eval_case_filter(case_filter)}"
        )

    suites = resolve_agent_eval_suites(suite_names, registry=registry)
    suite_reports = [_run_single_suite(suite, selected_cases) for suite in suites]
    failed_suite_count = sum(1 for report in suite_reports if not report.passed)

    return AgentEvalRunReport(
        cases_path=cases_path,
        selected_case_count=len(selected_cases),
        case_filter=describe_agent_eval_case_filter(case_filter),
        suite_count=len(suite_reports),
        passed_suite_count=len(suite_reports) - failed_suite_count,
        failed_suite_count=failed_suite_count,
        passed=failed_suite_count == 0,
        suite_reports=suite_reports,
    )


def format_agent_eval_run_report(report: AgentEvalRunReport) -> list[str]:
    lines = [
        "Agent evaluation suite",
        f"cases_path: {report.cases_path}",
        f"case_filter: {report.case_filter}",
        f"selected_cases: {report.selected_case_count}",
        f"suites: {', '.join(suite.name for suite in report.suite_reports)}",
        "",
    ]

    for index, suite_report in enumerate(report.suite_reports):
        if index > 0:
            lines.append("")
        lines.extend(_format_single_suite_report(suite_report))

    lines.extend(
        [
            "",
            "Overall",
            f"suites: {report.suite_count}",
            f"passed_suites: {report.passed_suite_count}",
            f"failed_suites: {report.failed_suite_count}",
            f"passed: {str(report.passed).lower()}",
        ]
    )
    return lines


def exit_code_for_agent_eval_report(report: AgentEvalRunReport) -> int:
    return 0 if report.passed else 1


def filter_agent_eval_cases(
    cases: Sequence[AgentEvalCase],
    *,
    case_filter: AgentEvalCaseFilter | None = None,
) -> list[AgentEvalCase]:
    if case_filter is None:
        return list(cases)

    include_tags = set(case_filter.include_tags)
    selected: list[AgentEvalCase] = []
    for eval_case in cases:
        if case_filter.priority is not None and eval_case.metadata.priority != case_filter.priority:
            continue
        if include_tags and not include_tags.issubset(set(eval_case.metadata.tags)):
            continue
        selected.append(eval_case)
    return selected


def describe_agent_eval_case_filter(
    case_filter: AgentEvalCaseFilter | None = None,
) -> str:
    if case_filter is None or (
        not case_filter.include_tags and case_filter.priority is None
    ):
        return "all"

    parts: list[str] = []
    if case_filter.include_tags:
        parts.append("tags=" + ",".join(case_filter.include_tags))
    if case_filter.priority is not None:
        parts.append(f"priority={case_filter.priority}")
    return ";".join(parts)


def _run_single_suite(
    suite: AgentEvalSuite,
    cases: Sequence[AgentEvalCase],
) -> AgentEvalSuiteReport:
    summary = suite.evaluate(cases)
    failed_case_count = _summary_int(summary, "failed_case_count")
    case_count = _summary_int(summary, "case_count")

    return AgentEvalSuiteReport(
        name=suite.name,
        title=suite.title,
        case_count=case_count,
        failed_case_count=failed_case_count,
        passed=failed_case_count == 0,
        summary_lines=suite.format_summary(summary),
        bad_case_lines=suite.format_bad_cases(summary),
    )


def _format_single_suite_report(report: AgentEvalSuiteReport) -> list[str]:
    return [
        f"== {report.name}: {report.title} ==",
        *report.summary_lines,
        *report.bad_case_lines,
    ]


def _summary_int(summary: EvaluationSummary, field_name: str) -> int:
    value = getattr(summary, field_name, None)
    if not isinstance(value, int):
        raise TypeError(f"Evaluation summary must expose integer field {field_name}")
    return value
