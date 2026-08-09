from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.agents.intent_evaluation import AgentEvalCase, AgentEvalPriority
from app.agents.ticket_agent import run_ticket_agent


AgentRunner = Callable[[str], Mapping[str, Any]]


class RagAgentEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected_expect_context: bool
    expected_rag_answer_status: str = Field(min_length=1)
    actual_rag_answer_status: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    actual_sources: list[str] = Field(default_factory=list)
    matched_sources: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    unexpected_sources: list[str] = Field(default_factory=list)
    expected_must_cite: bool
    citations_present: bool
    expected_should_create_ticket: bool
    actual_should_create_ticket: bool
    expected_ticket_need_source: str | None = None
    actual_ticket_need_source: str | None = None
    expected_issue_type: str | None = None
    actual_issue_type: str | None = None
    expected_confirmation_required: bool | None = None
    actual_confirmation_required: bool
    node_history: list[str] = Field(default_factory=list)
    rag_status_passed: bool
    sources_passed: bool
    citation_passed: bool
    ticket_decision_passed: bool
    no_context_behavior_passed: bool
    priority: AgentEvalPriority
    task_type: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    case_type: str = Field(min_length=1)
    passed: bool
    failed_reasons: list[str] = Field(default_factory=list)


class RagAgentEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    case_pass_rate: float = Field(ge=0, le=1)
    answered_case_count: int = Field(ge=0)
    answered_passed_case_count: int = Field(ge=0)
    no_context_case_count: int = Field(ge=0)
    no_context_passed_case_count: int = Field(ge=0)
    expected_source_count: int = Field(ge=0)
    matched_source_count: int = Field(ge=0)
    source_recall: float = Field(ge=0, le=1)
    citation_passed_count: int = Field(ge=0)
    ticket_decision_passed_count: int = Field(ge=0)
    p0_case_count: int = Field(ge=0)
    p0_passed_case_count: int = Field(ge=0)
    p0_failed_case_count: int = Field(ge=0)
    p0_case_pass_rate: float = Field(ge=0, le=1)
    results: list[RagAgentEvalCaseResult] = Field(default_factory=list)


def select_rag_agent_eval_cases(cases: Sequence[AgentEvalCase]) -> list[AgentEvalCase]:
    return [eval_case for eval_case in cases if isinstance(eval_case.expected.rag, dict)]


def evaluate_rag_agent_case(
    eval_case: AgentEvalCase,
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> RagAgentEvalCaseResult:
    expected_rag = _expected_rag(eval_case)
    if not expected_rag:
        raise ValueError("RAG + Agent eval case must define expected.rag")

    expected_ticket = _expected_ticket(eval_case)
    actual_state = dict(agent_runner(eval_case.inputs.message))
    expected_expect_context = expected_rag.get("expect_context") is True
    expected_rag_answer_status = "answered" if expected_expect_context else "no_context"
    actual_rag_answer_status = _optional_string(actual_state.get("rag_answer_status"))
    expected_sources = _string_list(expected_rag.get("expected_sources"))
    actual_sources = _citation_sources(actual_state.get("rag_citations"))
    matched_sources = [
        source for source in expected_sources if source in set(actual_sources)
    ]
    missing_sources = [
        source for source in expected_sources if source not in set(actual_sources)
    ]
    unexpected_sources = [
        source for source in actual_sources if source not in set(expected_sources)
    ]
    expected_must_cite = expected_rag.get("must_cite") is True
    citations_present = bool(actual_sources)
    expected_should_create_ticket = expected_ticket.get("should_create_ticket") is True
    actual_should_create_ticket = _actual_should_create_ticket(actual_state)
    expected_ticket_need_source = _optional_string(
        expected_ticket.get("ticket_need_source")
    )
    actual_ticket_need_source = _optional_string(actual_state.get("ticket_need_source"))
    expected_issue_type = _optional_string(expected_ticket.get("expected_issue_type"))
    actual_issue_type = _actual_issue_type(actual_state)
    expected_confirmation_required = _optional_bool(
        expected_ticket.get("confirmation_required")
    )
    actual_confirmation_required = actual_state.get("ticket_confirmation_required") is True
    rag_status_passed = actual_rag_answer_status == expected_rag_answer_status
    sources_passed = (
        not missing_sources
        and (expected_expect_context or not actual_sources)
    )
    citation_passed = not expected_must_cite or citations_present
    ticket_decision_passed = actual_should_create_ticket == expected_should_create_ticket
    no_context_behavior_passed = (
        True
        if expected_expect_context
        else (
            actual_rag_answer_status == "no_context"
            and not actual_sources
            and actual_should_create_ticket is True
            and actual_ticket_need_source == "rag_no_context"
        )
    )
    failed_reasons = _collect_failed_reasons(
        expected_rag_answer_status=expected_rag_answer_status,
        actual_rag_answer_status=actual_rag_answer_status,
        missing_sources=missing_sources,
        unexpected_sources=unexpected_sources,
        expected_must_cite=expected_must_cite,
        citations_present=citations_present,
        expected_should_create_ticket=expected_should_create_ticket,
        actual_should_create_ticket=actual_should_create_ticket,
        expected_ticket_need_source=expected_ticket_need_source,
        actual_ticket_need_source=actual_ticket_need_source,
        expected_issue_type=expected_issue_type,
        actual_issue_type=actual_issue_type,
        expected_confirmation_required=expected_confirmation_required,
        actual_confirmation_required=actual_confirmation_required,
        no_context_behavior_passed=no_context_behavior_passed,
    )

    return RagAgentEvalCaseResult(
        case_id=eval_case.id,
        name=eval_case.name,
        message=eval_case.inputs.message,
        expected_expect_context=expected_expect_context,
        expected_rag_answer_status=expected_rag_answer_status,
        actual_rag_answer_status=actual_rag_answer_status,
        expected_sources=expected_sources,
        actual_sources=actual_sources,
        matched_sources=matched_sources,
        missing_sources=missing_sources,
        unexpected_sources=unexpected_sources,
        expected_must_cite=expected_must_cite,
        citations_present=citations_present,
        expected_should_create_ticket=expected_should_create_ticket,
        actual_should_create_ticket=actual_should_create_ticket,
        expected_ticket_need_source=expected_ticket_need_source,
        actual_ticket_need_source=actual_ticket_need_source,
        expected_issue_type=expected_issue_type,
        actual_issue_type=actual_issue_type,
        expected_confirmation_required=expected_confirmation_required,
        actual_confirmation_required=actual_confirmation_required,
        node_history=_string_list(actual_state.get("node_history")),
        rag_status_passed=rag_status_passed,
        sources_passed=sources_passed,
        citation_passed=citation_passed,
        ticket_decision_passed=ticket_decision_passed,
        no_context_behavior_passed=no_context_behavior_passed,
        priority=eval_case.metadata.priority,
        task_type=eval_case.metadata.task_type,
        business_domain=eval_case.metadata.business_domain,
        case_type=eval_case.metadata.case_type,
        passed=not failed_reasons,
        failed_reasons=failed_reasons,
    )


def evaluate_rag_agent_cases(
    cases: Sequence[AgentEvalCase],
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> RagAgentEvalSummary:
    eval_cases = select_rag_agent_eval_cases(cases)
    results = [
        evaluate_rag_agent_case(eval_case, agent_runner=agent_runner)
        for eval_case in eval_cases
    ]
    answered_results = [
        result for result in results if result.expected_rag_answer_status == "answered"
    ]
    no_context_results = [
        result for result in results if result.expected_rag_answer_status == "no_context"
    ]
    p0_results = [result for result in results if result.priority == "p0"]
    expected_source_count = sum(len(result.expected_sources) for result in results)
    matched_source_count = sum(len(result.matched_sources) for result in results)

    return RagAgentEvalSummary(
        case_count=len(results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        case_pass_rate=_ratio(
            sum(1 for result in results if result.passed),
            len(results),
        ),
        answered_case_count=len(answered_results),
        answered_passed_case_count=sum(
            1 for result in answered_results if result.passed
        ),
        no_context_case_count=len(no_context_results),
        no_context_passed_case_count=sum(
            1 for result in no_context_results if result.passed
        ),
        expected_source_count=expected_source_count,
        matched_source_count=matched_source_count,
        source_recall=_ratio(matched_source_count, expected_source_count),
        citation_passed_count=sum(1 for result in results if result.citation_passed),
        ticket_decision_passed_count=sum(
            1 for result in results if result.ticket_decision_passed
        ),
        p0_case_count=len(p0_results),
        p0_passed_case_count=sum(1 for result in p0_results if result.passed),
        p0_failed_case_count=sum(1 for result in p0_results if not result.passed),
        p0_case_pass_rate=_ratio(
            sum(1 for result in p0_results if result.passed),
            len(p0_results),
        ),
        results=results,
    )


def format_rag_agent_eval_summary(summary: RagAgentEvalSummary) -> list[str]:
    return [
        "RAG + Agent evaluation summary",
        f"cases: {summary.case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"case_pass_rate: {summary.case_pass_rate:.4f}",
        f"answered_cases: {summary.answered_case_count}",
        f"answered_passed_cases: {summary.answered_passed_case_count}",
        f"no_context_cases: {summary.no_context_case_count}",
        f"no_context_passed_cases: {summary.no_context_passed_case_count}",
        f"expected_sources: {summary.expected_source_count}",
        f"matched_sources: {summary.matched_source_count}",
        f"source_recall: {summary.source_recall:.4f}",
        f"citation_passed_count: {summary.citation_passed_count}",
        f"ticket_decision_passed_count: {summary.ticket_decision_passed_count}",
        f"p0_cases: {summary.p0_case_count}",
        f"p0_passed_cases: {summary.p0_passed_case_count}",
        f"p0_failed_cases: {summary.p0_failed_case_count}",
        f"p0_case_pass_rate: {summary.p0_case_pass_rate:.4f}",
    ]


def format_rag_agent_bad_cases(summary: RagAgentEvalSummary) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["Bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: expected_status={result.expected_rag_answer_status} "
            f"actual_status={result.actual_rag_answer_status or '-'} "
            f"priority={result.priority}"
        )
        lines.append(f"  expected_sources: {result.expected_sources}")
        lines.append(f"  actual_sources: {result.actual_sources}")
        for reason in result.failed_reasons:
            lines.append(f"  - {reason}")
    return lines


def _expected_rag(eval_case: AgentEvalCase) -> dict[str, Any]:
    rag = eval_case.expected.rag
    if not isinstance(rag, dict):
        return {}
    return rag


def _expected_ticket(eval_case: AgentEvalCase) -> dict[str, Any]:
    ticket = eval_case.expected.ticket
    if not isinstance(ticket, dict):
        return {}
    return ticket


def _citation_sources(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    sources: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source = _optional_string(item.get("source"))
        if source is not None and source not in sources:
            sources.append(source)
    return sources


def _actual_should_create_ticket(actual_state: Mapping[str, Any]) -> bool:
    return actual_state.get("needs_ticket") is True or bool(
        actual_state.get("ticket_fields")
    )


def _actual_issue_type(actual_state: Mapping[str, Any]) -> str | None:
    fields = actual_state.get("ticket_fields")
    if not isinstance(fields, Mapping):
        return None
    return _optional_string(fields.get("issue_type"))


def _collect_failed_reasons(
    *,
    expected_rag_answer_status: str,
    actual_rag_answer_status: str | None,
    missing_sources: list[str],
    unexpected_sources: list[str],
    expected_must_cite: bool,
    citations_present: bool,
    expected_should_create_ticket: bool,
    actual_should_create_ticket: bool,
    expected_ticket_need_source: str | None,
    actual_ticket_need_source: str | None,
    expected_issue_type: str | None,
    actual_issue_type: str | None,
    expected_confirmation_required: bool | None,
    actual_confirmation_required: bool,
    no_context_behavior_passed: bool,
) -> list[str]:
    reasons: list[str] = []
    if actual_rag_answer_status != expected_rag_answer_status:
        reasons.append(
            "rag_answer_status expected="
            f"{expected_rag_answer_status!r} actual={actual_rag_answer_status!r}"
        )
    if missing_sources:
        reasons.append(f"missing_sources={missing_sources!r}")
    if expected_rag_answer_status == "no_context" and unexpected_sources:
        reasons.append(f"unexpected_sources={unexpected_sources!r}")
    if expected_must_cite and not citations_present:
        reasons.append("expected citations but actual citations are empty")
    if actual_should_create_ticket != expected_should_create_ticket:
        reasons.append(
            "should_create_ticket expected="
            f"{expected_should_create_ticket!r} actual={actual_should_create_ticket!r}"
        )
    if (
        expected_ticket_need_source is not None
        and actual_ticket_need_source != expected_ticket_need_source
    ):
        reasons.append(
            "ticket_need_source expected="
            f"{expected_ticket_need_source!r} actual={actual_ticket_need_source!r}"
        )
    if expected_issue_type is not None and actual_issue_type != expected_issue_type:
        reasons.append(
            f"issue_type expected={expected_issue_type!r} actual={actual_issue_type!r}"
        )
    if (
        expected_confirmation_required is not None
        and actual_confirmation_required != expected_confirmation_required
    ):
        reasons.append(
            "confirmation_required expected="
            f"{expected_confirmation_required!r} actual={actual_confirmation_required!r}"
        )
    if expected_rag_answer_status == "no_context" and not no_context_behavior_passed:
        reasons.append("no_context behavior did not route to rag_no_context ticket")
    return reasons


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("expected a list of strings")
    return [item for item in value if isinstance(item, str) and item.strip()]


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
