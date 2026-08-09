from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.agents.intent_evaluation import AgentEvalCase, AgentEvalPriority
from app.agents.ticket_agent import run_ticket_agent


AgentRunner = Callable[[str], Mapping[str, Any]]


class TicketFieldComparison(BaseModel):
    field_name: str = Field(min_length=1)
    expected_value: Any
    actual_value: Any = None
    passed: bool
    failed_reason: str | None = None


class TicketFieldEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected_should_create_ticket: bool
    actual_should_create_ticket: bool
    expected_ticket_need_source: str | None = None
    actual_ticket_need_source: str | None = None
    expected_missing_ticket_fields: list[str] = Field(default_factory=list)
    actual_missing_ticket_fields: list[str] = Field(default_factory=list)
    expected_confirmation_required: bool | None = None
    actual_confirmation_required: bool
    expected_fields: dict[str, Any] = Field(default_factory=dict)
    actual_fields: dict[str, Any] = Field(default_factory=dict)
    field_comparisons: list[TicketFieldComparison] = Field(default_factory=list)
    expected_field_count: int = Field(ge=0)
    matched_field_count: int = Field(ge=0)
    field_accuracy: float = Field(ge=0, le=1)
    priority: AgentEvalPriority
    task_type: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    case_type: str = Field(min_length=1)
    passed: bool
    failed_reasons: list[str] = Field(default_factory=list)


class TicketFieldEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    case_pass_rate: float = Field(ge=0, le=1)
    expected_field_count: int = Field(ge=0)
    matched_field_count: int = Field(ge=0)
    field_accuracy: float = Field(ge=0, le=1)
    p0_case_count: int = Field(ge=0)
    p0_passed_case_count: int = Field(ge=0)
    p0_failed_case_count: int = Field(ge=0)
    p0_case_pass_rate: float = Field(ge=0, le=1)
    missing_field_case_count: int = Field(ge=0)
    missing_field_passed_case_count: int = Field(ge=0)
    results: list[TicketFieldEvalCaseResult] = Field(default_factory=list)


def select_ticket_field_eval_cases(cases: Sequence[AgentEvalCase]) -> list[AgentEvalCase]:
    return [
        eval_case
        for eval_case in cases
        if _expected_ticket(eval_case).get("should_create_ticket") is True
    ]


def evaluate_ticket_field_case(
    eval_case: AgentEvalCase,
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> TicketFieldEvalCaseResult:
    expected_ticket = _expected_ticket(eval_case)
    if expected_ticket.get("should_create_ticket") is not True:
        raise ValueError("ticket field eval case must expect ticket creation")

    actual_state = dict(agent_runner(eval_case.inputs.message))
    expected_fields = _expected_fields(expected_ticket)
    actual_fields = _actual_ticket_fields(actual_state)
    field_comparisons = [
        _compare_field(
            field_name=field_name,
            expected_value=expected_value,
            actual_fields=actual_fields,
        )
        for field_name, expected_value in expected_fields.items()
    ]
    expected_missing_fields = _string_list(expected_ticket.get("missing_ticket_fields"))
    actual_missing_fields = _string_list(actual_state.get("missing_ticket_fields"))
    expected_confirmation_required = _optional_bool(
        expected_ticket.get("confirmation_required")
    )
    actual_confirmation_required = actual_state.get("ticket_confirmation_required") is True
    failed_reasons = _collect_failed_reasons(
        expected_should_create_ticket=True,
        actual_should_create_ticket=_actual_should_create_ticket(actual_state),
        expected_ticket_need_source=_optional_string(
            expected_ticket.get("ticket_need_source")
        ),
        actual_ticket_need_source=_optional_string(
            actual_state.get("ticket_need_source")
        ),
        expected_missing_fields=expected_missing_fields,
        actual_missing_fields=actual_missing_fields,
        expected_confirmation_required=expected_confirmation_required,
        actual_confirmation_required=actual_confirmation_required,
        field_comparisons=field_comparisons,
    )
    matched_field_count = sum(1 for comparison in field_comparisons if comparison.passed)

    return TicketFieldEvalCaseResult(
        case_id=eval_case.id,
        name=eval_case.name,
        message=eval_case.inputs.message,
        expected_should_create_ticket=True,
        actual_should_create_ticket=_actual_should_create_ticket(actual_state),
        expected_ticket_need_source=_optional_string(
            expected_ticket.get("ticket_need_source")
        ),
        actual_ticket_need_source=_optional_string(
            actual_state.get("ticket_need_source")
        ),
        expected_missing_ticket_fields=expected_missing_fields,
        actual_missing_ticket_fields=actual_missing_fields,
        expected_confirmation_required=expected_confirmation_required,
        actual_confirmation_required=actual_confirmation_required,
        expected_fields=expected_fields,
        actual_fields=actual_fields,
        field_comparisons=field_comparisons,
        expected_field_count=len(field_comparisons),
        matched_field_count=matched_field_count,
        field_accuracy=_ratio(matched_field_count, len(field_comparisons)),
        priority=eval_case.metadata.priority,
        task_type=eval_case.metadata.task_type,
        business_domain=eval_case.metadata.business_domain,
        case_type=eval_case.metadata.case_type,
        passed=not failed_reasons,
        failed_reasons=failed_reasons,
    )


def evaluate_ticket_field_cases(
    cases: Sequence[AgentEvalCase],
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> TicketFieldEvalSummary:
    eval_cases = select_ticket_field_eval_cases(cases)
    results = [
        evaluate_ticket_field_case(eval_case, agent_runner=agent_runner)
        for eval_case in eval_cases
    ]
    p0_results = [result for result in results if result.priority == "p0"]
    missing_field_results = [
        result for result in results if result.expected_missing_ticket_fields
    ]
    expected_field_count = sum(result.expected_field_count for result in results)
    matched_field_count = sum(result.matched_field_count for result in results)

    return TicketFieldEvalSummary(
        case_count=len(results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        case_pass_rate=_ratio(
            sum(1 for result in results if result.passed),
            len(results),
        ),
        expected_field_count=expected_field_count,
        matched_field_count=matched_field_count,
        field_accuracy=_ratio(matched_field_count, expected_field_count),
        p0_case_count=len(p0_results),
        p0_passed_case_count=sum(1 for result in p0_results if result.passed),
        p0_failed_case_count=sum(1 for result in p0_results if not result.passed),
        p0_case_pass_rate=_ratio(
            sum(1 for result in p0_results if result.passed),
            len(p0_results),
        ),
        missing_field_case_count=len(missing_field_results),
        missing_field_passed_case_count=sum(
            1 for result in missing_field_results if result.passed
        ),
        results=results,
    )


def format_ticket_field_eval_summary(summary: TicketFieldEvalSummary) -> list[str]:
    return [
        "Agent ticket field evaluation summary",
        f"cases: {summary.case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"case_pass_rate: {summary.case_pass_rate:.4f}",
        f"expected_fields: {summary.expected_field_count}",
        f"matched_fields: {summary.matched_field_count}",
        f"field_accuracy: {summary.field_accuracy:.4f}",
        f"p0_cases: {summary.p0_case_count}",
        f"p0_passed_cases: {summary.p0_passed_case_count}",
        f"p0_failed_cases: {summary.p0_failed_case_count}",
        f"p0_case_pass_rate: {summary.p0_case_pass_rate:.4f}",
        f"missing_field_cases: {summary.missing_field_case_count}",
        (
            "missing_field_passed_cases: "
            f"{summary.missing_field_passed_case_count}"
        ),
    ]


def format_ticket_field_bad_cases(summary: TicketFieldEvalSummary) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["Bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: priority={result.priority} "
            f"task_type={result.task_type} field_accuracy={result.field_accuracy:.4f}"
        )
        for reason in result.failed_reasons:
            lines.append(f"  - {reason}")
    return lines


def _expected_ticket(eval_case: AgentEvalCase) -> dict[str, Any]:
    ticket = eval_case.expected.ticket
    if not isinstance(ticket, dict):
        return {}
    return ticket


def _expected_fields(expected_ticket: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    expected_fields = expected_ticket.get("expected_fields")
    if isinstance(expected_fields, Mapping):
        fields.update(dict(expected_fields))

    expected_issue_type = expected_ticket.get("expected_issue_type")
    if "issue_type" not in fields and expected_issue_type is not None:
        fields["issue_type"] = expected_issue_type

    return fields


def _actual_ticket_fields(actual_state: Mapping[str, Any]) -> dict[str, Any]:
    fields = actual_state.get("ticket_fields")
    if not isinstance(fields, Mapping):
        return {}
    return dict(fields)


def _actual_should_create_ticket(actual_state: Mapping[str, Any]) -> bool:
    return actual_state.get("needs_ticket") is True or bool(
        actual_state.get("ticket_fields")
    )


def _compare_field(
    *,
    field_name: str,
    expected_value: Any,
    actual_fields: Mapping[str, Any],
) -> TicketFieldComparison:
    actual_value = actual_fields.get(field_name)
    passed = actual_value == expected_value
    return TicketFieldComparison(
        field_name=field_name,
        expected_value=expected_value,
        actual_value=actual_value,
        passed=passed,
        failed_reason=None
        if passed
        else (
            f"field {field_name!r} expected={expected_value!r} "
            f"actual={actual_value!r}"
        ),
    )


def _collect_failed_reasons(
    *,
    expected_should_create_ticket: bool,
    actual_should_create_ticket: bool,
    expected_ticket_need_source: str | None,
    actual_ticket_need_source: str | None,
    expected_missing_fields: list[str],
    actual_missing_fields: list[str],
    expected_confirmation_required: bool | None,
    actual_confirmation_required: bool,
    field_comparisons: Sequence[TicketFieldComparison],
) -> list[str]:
    reasons: list[str] = []
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
    if set(actual_missing_fields) != set(expected_missing_fields):
        reasons.append(
            f"missing_ticket_fields expected={expected_missing_fields!r} "
            f"actual={actual_missing_fields!r}"
        )
    if (
        expected_confirmation_required is not None
        and actual_confirmation_required != expected_confirmation_required
    ):
        reasons.append(
            "confirmation_required expected="
            f"{expected_confirmation_required!r} actual={actual_confirmation_required!r}"
        )
    for comparison in field_comparisons:
        if comparison.failed_reason is not None:
            reasons.append(comparison.failed_reason)
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
