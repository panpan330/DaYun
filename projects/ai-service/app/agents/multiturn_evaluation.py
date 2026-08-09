"""Multiturn conversation evaluation for the ticket agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.multiturn_eval import MultiturnEvalCase

# intent -> route 静态映射（与 ticket_agent.TICKET_AGENT_INTENT_ROUTES 一致）
_INTENT_TO_ROUTE: dict[str, str] = {
    "policy_question": "retrieve_policy",
    "order_query": "query_order",
    "ticket_request": "decide_ticket_need",
    "refund_request": "handle_refund_request",
    "cancel_request": "handle_cancel_request",
    "smalltalk": "build_direct_answer",
    "unsupported": "build_unsupported_answer",
    "unclear": "ask_clarifying_question",
}


class MultiturnTurnResult(BaseModel):
    turn_index: int = Field(ge=0)
    message: str
    expected_intent: str
    expected_route: str
    base_intent: str | None = None
    context_intent: str | None = None
    passed: bool
    reason: str


class MultiturnCaseResult(BaseModel):
    case_id: str
    case_name: str
    passed: bool
    turn_results: list[MultiturnTurnResult]
    final_answer_passed: bool | None = None


class MultiturnEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    bad_cases: list[MultiturnCaseResult] = Field(default_factory=list)


def _ratio(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def _intent_of(result: Any) -> str | None:
    if isinstance(result, dict):
        return result.get("intent")
    return getattr(result, "intent", None)


def _route_of(result: Any, intent: str | None) -> str | None:
    if isinstance(result, dict):
        return _INTENT_TO_ROUTE.get(intent or "")
    route = getattr(result, "intent_route", None) or getattr(result, "route", None)
    return route or _INTENT_TO_ROUTE.get(intent or "")


def evaluate_multiturn_cases(
    cases: Sequence[MultiturnEvalCase],
    *,
    classifier: Callable[..., Any] | None = None,
    strict_history: bool = False,
) -> MultiturnEvalSummary:
    """Evaluate multiturn cases.

    ``strict_history`` enables the cross-turn assertion: for ``requires_history``
    turns the base (no-history) classification must differ from the expectation
    and the context classification must match. The rule-based classifier is
    keyword-driven and does not differ with history, so strict mode is meant for
    LLM/injected classifiers; the suite runs with ``strict_history=False`` to
    stay deterministic offline.
    """
    if classifier is None:
        from app.agents.ticket_agent import RuleBasedTicketIntentClassifier

        classifier = RuleBasedTicketIntentClassifier().classify_intent

    results: list[MultiturnCaseResult] = []
    for case in cases:
        history: list[str] = []
        turn_results: list[MultiturnTurnResult] = []
        case_passed = True
        for index, turn in enumerate(case.conversation):
            base = classifier(turn.message)
            context = classifier(turn.message, history=history)
            base_intent = _intent_of(base)
            context_intent = _intent_of(context)
            context_route = _route_of(context, context_intent)

            reason = ""
            passed = True
            if context_intent != turn.expected_intent or context_route != turn.expected_route:
                passed = False
                reason = (
                    f"context intent/route mismatch: got {context_intent}/{context_route}"
                )
            if turn.requires_history and strict_history:
                if base_intent == turn.expected_intent:
                    passed = False
                    reason = (
                        "requires_history but base classification already matches "
                        "(assertion too loose)"
                    )
                elif not (
                    context_intent == turn.expected_intent
                    and context_route == turn.expected_route
                ):
                    passed = False
                    reason = "requires_history but context classification does not match"
            if passed:
                reason = "ok"
            turn_results.append(
                MultiturnTurnResult(
                    turn_index=index,
                    message=turn.message,
                    expected_intent=turn.expected_intent,
                    expected_route=turn.expected_route,
                    base_intent=base_intent,
                    context_intent=context_intent,
                    passed=passed,
                    reason=reason,
                )
            )
            if not passed:
                case_passed = False
            history.append(turn.message)

        final_passed: bool | None = None
        if case.final_answer is not None:
            final_context = classifier(
                case.conversation[-1].message,
                history=history[:-1],
            )
            final_intent = _intent_of(final_context)
            final_passed = final_intent == case.final_answer.expected_intent
            if not final_passed:
                case_passed = False

        results.append(
            MultiturnCaseResult(
                case_id=case.id,
                case_name=case.name,
                passed=case_passed,
                turn_results=turn_results,
                final_answer_passed=final_passed,
            )
        )

    passed_count = sum(1 for result in results if result.passed)
    return MultiturnEvalSummary(
        case_count=len(results),
        passed_case_count=passed_count,
        failed_case_count=len(results) - passed_count,
        accuracy=_ratio(passed_count, len(results)),
        bad_cases=[result for result in results if not result.passed],
    )


def format_multiturn_eval_summary(summary: MultiturnEvalSummary) -> list[str]:
    return [
        f"多轮评测：{summary.passed_case_count}/{summary.case_count} 通过",
        f"准确率：{summary.accuracy:.2%}",
    ]


def format_multiturn_bad_cases(summary: MultiturnEvalSummary) -> list[str]:
    lines: list[str] = []
    for result in summary.bad_cases:
        lines.append(f"[{result.case_id}] {result.case_name}")
        for turn in result.turn_results:
            if not turn.passed:
                lines.append(
                    f"  第 {turn.turn_index + 1} 轮：期望 {turn.expected_intent}/{turn.expected_route}"
                    f"，实际 context {turn.context_intent}；{turn.reason}"
                )
        if result.final_answer_passed is False:
            lines.append("  末轮 final_answer 断言未通过")
    return lines
