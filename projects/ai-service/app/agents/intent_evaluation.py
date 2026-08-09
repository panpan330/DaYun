from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agents.ticket_agent import (
    TICKET_AGENT_INTENT_ROUTES,
    TicketIntent,
    classify_ticket_intent,
)


AgentEvalPriority = Literal["p0", "p1", "p2"]


class AgentEvalInputs(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class AgentEvalExpected(BaseModel):
    intent: TicketIntent
    intent_route: str = Field(min_length=1)
    rag: dict[str, Any] | None = None
    ticket: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    must_ask_for: list[str] = Field(default_factory=list)
    must_not_reveal: list[str] = Field(default_factory=list)
    emotion: str | None = None

    @model_validator(mode="after")
    def validate_intent_route(self) -> "AgentEvalExpected":
        expected_route = TICKET_AGENT_INTENT_ROUTES[self.intent]
        if self.intent_route != expected_route:
            raise ValueError(
                f"intent_route must be {expected_route!r} for intent {self.intent!r}"
            )
        return self


class AgentEvalMetadata(BaseModel):
    task_type: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    case_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    priority: AgentEvalPriority
    tags: list[str] = Field(default_factory=list)

    @field_validator(
        "task_type",
        "business_domain",
        "case_type",
        "difficulty",
        mode="before",
    )
    @classmethod
    def normalize_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("tags must be a list of strings")
        normalized_tags: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("tags must contain non-blank strings")
            tag = item.strip()
            if tag not in normalized_tags:
                normalized_tags.append(tag)
        return normalized_tags


class AgentEvalCase(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    inputs: AgentEvalInputs
    expected: AgentEvalExpected
    metadata: AgentEvalMetadata

    @field_validator("id", "name", mode="before")
    @classmethod
    def normalize_required_string(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class AgentEvalDataset(BaseModel):
    schema_version: str = Field(min_length=1)
    description: str = ""
    cases: list[AgentEvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "AgentEvalDataset":
        _validate_unique_case_ids(self.cases)
        return self


class IntentEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected_intent: str = Field(min_length=1)
    actual_intent: str = Field(min_length=1)
    expected_route: str = Field(min_length=1)
    actual_route: str | None = None
    classifier_reason: str = ""
    priority: AgentEvalPriority
    task_type: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    case_type: str = Field(min_length=1)
    passed: bool
    failed_reason: str | None = None


class IntentEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    p0_case_count: int = Field(ge=0)
    p0_passed_case_count: int = Field(ge=0)
    p0_failed_case_count: int = Field(ge=0)
    p0_accuracy: float = Field(ge=0, le=1)
    results: list[IntentEvalCaseResult] = Field(default_factory=list)


IntentClassifier = Callable[[str], Mapping[str, str]]


def load_agent_eval_dataset(path: Path | str) -> AgentEvalDataset:
    raw_text = Path(path).read_text(encoding="utf-8")
    raw_dataset = json.loads(raw_text)
    return AgentEvalDataset.model_validate(raw_dataset)


def load_agent_eval_cases(path: Path | str) -> list[AgentEvalCase]:
    return load_agent_eval_dataset(path).cases


def evaluate_intent_case(
    eval_case: AgentEvalCase,
    *,
    classifier: IntentClassifier = classify_ticket_intent,
) -> IntentEvalCaseResult:
    classification = classifier(eval_case.inputs.message)
    actual_intent = _get_classifier_value(classification, "intent")
    classifier_reason = _get_classifier_value(classification, "reason")
    actual_route = TICKET_AGENT_INTENT_ROUTES.get(actual_intent)
    expected_intent = eval_case.expected.intent
    expected_route = eval_case.expected.intent_route
    passed = actual_intent == expected_intent and actual_route == expected_route

    return IntentEvalCaseResult(
        case_id=eval_case.id,
        name=eval_case.name,
        message=eval_case.inputs.message,
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        expected_route=expected_route,
        actual_route=actual_route,
        classifier_reason=classifier_reason,
        priority=eval_case.metadata.priority,
        task_type=eval_case.metadata.task_type,
        business_domain=eval_case.metadata.business_domain,
        case_type=eval_case.metadata.case_type,
        passed=passed,
        failed_reason=None
        if passed
        else (
            f"expected intent={expected_intent} route={expected_route}; "
            f"got intent={actual_intent} route={actual_route or '-'}"
        ),
    )


def evaluate_intent_cases(
    cases: Sequence[AgentEvalCase],
    *,
    classifier: IntentClassifier = classify_ticket_intent,
) -> IntentEvalSummary:
    _validate_unique_case_ids(cases)
    results = [
        evaluate_intent_case(eval_case, classifier=classifier)
        for eval_case in cases
    ]
    p0_results = [result for result in results if result.priority == "p0"]

    return IntentEvalSummary(
        case_count=len(results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        accuracy=_ratio(
            sum(1 for result in results if result.passed),
            len(results),
        ),
        p0_case_count=len(p0_results),
        p0_passed_case_count=sum(1 for result in p0_results if result.passed),
        p0_failed_case_count=sum(1 for result in p0_results if not result.passed),
        p0_accuracy=_ratio(
            sum(1 for result in p0_results if result.passed),
            len(p0_results),
        ),
        results=results,
    )


def format_intent_eval_summary(summary: IntentEvalSummary) -> list[str]:
    return [
        "Agent intent evaluation summary",
        f"cases: {summary.case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"accuracy: {summary.accuracy:.4f}",
        f"p0_cases: {summary.p0_case_count}",
        f"p0_passed_cases: {summary.p0_passed_case_count}",
        f"p0_failed_cases: {summary.p0_failed_case_count}",
        f"p0_accuracy: {summary.p0_accuracy:.4f}",
    ]


def format_intent_bad_cases(summary: IntentEvalSummary) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["Bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: expected={result.expected_intent} "
            f"actual={result.actual_intent} priority={result.priority} "
            f"task_type={result.task_type} reason={result.failed_reason}"
        )
    return lines


def _get_classifier_value(classification: Mapping[str, str], key: str) -> str:
    value = classification.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"classifier result field {key!r} must be a string")
    normalized = value.strip()
    if key == "intent" and not normalized:
        raise ValueError("classifier result must contain a non-blank intent")
    return normalized


def _validate_unique_case_ids(cases: Sequence[AgentEvalCase]) -> None:
    seen: set[str] = set()
    for eval_case in cases:
        if eval_case.id in seen:
            raise ValueError("agent eval case ids must be unique")
        seen.add(eval_case.id)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
