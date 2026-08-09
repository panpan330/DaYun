from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agents.must_check import check_must_ask_for, check_must_not_reveal
from app.agents.ticket_agent import run_ticket_agent
from app.evaluation.bad_case_registry import BadCaseRecord


ProductionRegressionOutcome = Literal["passed", "failed", "not_ready", "error"]
AgentRunner = Callable[[str], Mapping[str, Any]]

# tool_called 断言：supervisor 按工具名填 expected_tool，但 node_history 记录的是
# 真实执行节点名，二者并不总是一致（如退款工具 refund_order 由 execute_refund_request
# 节点执行）。这里维护 工具名 → 节点名 的别名表，匹配时把工具名与节点名都纳入候选，
# 填 `refund_order` 或 `execute_refund_request` 均能命中退款执行节点。
_TOOL_CALLED_NODE_ALIASES: Mapping[str, str] = {
    "refund_order": "execute_refund_request",
}


class ProductionRegressionCaseResult(BaseModel):
    bad_case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    outcome: ProductionRegressionOutcome
    assertion: str | None = None
    expected: str | None = None
    actual: str | None = None
    detail: str = Field(min_length=1)


class ProductionRegressionRun(BaseModel):
    run_id: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime
    total_case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    not_ready_case_count: int = Field(ge=0)
    error_case_count: int = Field(ge=0)
    passed: bool
    results: list[ProductionRegressionCaseResult] = Field(default_factory=list)


def run_production_bad_case_regression(
    records: Sequence[BadCaseRecord],
    *,
    agent_runner: AgentRunner = run_ticket_agent,
    now: Callable[[], datetime] | None = None,
) -> ProductionRegressionRun:
    clock = now or (lambda: datetime.now(timezone.utc))
    started_at = clock()
    regression_records = [
        record
        for record in records
        if record.source == "production" and record.status == "regression_added"
    ]
    results = [
        _run_single_production_bad_case(record, agent_runner=agent_runner)
        for record in regression_records
    ]
    passed_count = sum(result.outcome == "passed" for result in results)
    failed_count = sum(result.outcome == "failed" for result in results)
    not_ready_count = sum(result.outcome == "not_ready" for result in results)
    error_count = sum(result.outcome == "error" for result in results)

    return ProductionRegressionRun(
        run_id=f"production-regression-{uuid4().hex}",
        started_at=started_at,
        completed_at=clock(),
        total_case_count=len(results),
        passed_case_count=passed_count,
        failed_case_count=failed_count,
        not_ready_case_count=not_ready_count,
        error_case_count=error_count,
        passed=bool(results) and failed_count == 0 and not_ready_count == 0 and error_count == 0,
        results=results,
    )


def _run_single_production_bad_case(
    record: BadCaseRecord,
    *,
    agent_runner: AgentRunner,
) -> ProductionRegressionCaseResult:
    spec = record.production_regression
    if spec is None:
        return ProductionRegressionCaseResult(
            bad_case_id=record.id,
            title=record.title,
            outcome="not_ready",
            detail="This formal bad case has no supervisor-defined executable regression assertion.",
        )

    try:
        state = dict(agent_runner(spec.message))
    except Exception as exc:  # The run must retain one result instead of aborting every case.
        return ProductionRegressionCaseResult(
            bad_case_id=record.id,
            title=record.title,
            outcome="error",
            assertion=spec.assertion,
            detail=f"Agent execution failed: {type(exc).__name__}.",
        )

    if spec.assertion == "intent":
        actual = _optional_string(state.get("intent"))
        expected = spec.expected_intent
        return _assert_result(
            record,
            assertion="intent",
            expected=expected,
            actual=actual,
            passed=actual == expected,
        )
    if spec.assertion == "citation_present":
        citations = state.get("rag_citations")
        actual = "present" if isinstance(citations, list) and bool(citations) else "missing"
        return _assert_result(
            record,
            assertion="citation_present",
            expected="present",
            actual=actual,
            passed=actual == "present",
        )
    if spec.assertion == "ticket_confirmation_required":
        actual = "required" if state.get("ticket_confirmation_required") is True else "not_required"
        return _assert_result(
            record,
            assertion="ticket_confirmation_required",
            expected="required",
            actual=actual,
            passed=actual == "required",
        )
    if spec.assertion == "tool_called":
        node_history = state.get("node_history") or []
        candidates = {spec.expected_tool}
        node_alias = _TOOL_CALLED_NODE_ALIASES.get(spec.expected_tool)
        if node_alias:
            candidates.add(node_alias)
        matched = [
            str(node)
            for node in node_history
            if any(candidate in str(node) for candidate in candidates)
        ]
        return _assert_result(
            record,
            assertion="tool_called",
            expected=spec.expected_tool,
            actual=", ".join(matched) if matched else "not_called",
            passed=bool(matched),
        )
    if spec.assertion == "must_ask_for":
        reply = str(state.get("final_answer") or "")
        missing = check_must_ask_for(reply, spec.must_ask_fields)
        return _assert_result(
            record,
            assertion="must_ask_for",
            expected=", ".join(spec.must_ask_fields),
            actual=", ".join(missing) if missing else "all_asked",
            passed=not missing,
        )
    if spec.assertion == "must_not_reveal":
        reply = str(state.get("final_answer") or "")
        leaked = check_must_not_reveal(reply, spec.must_not_reveal_terms)
        return _assert_result(
            record,
            assertion="must_not_reveal",
            expected="none",
            actual=", ".join(leaked) if leaked else "none_revealed",
            passed=not leaked,
        )

    raise ValueError(f"unsupported production regression assertion: {spec.assertion}")


def _assert_result(
    record: BadCaseRecord,
    *,
    assertion: str,
    expected: str | None,
    actual: str | None,
    passed: bool,
) -> ProductionRegressionCaseResult:
    return ProductionRegressionCaseResult(
        bad_case_id=record.id,
        title=record.title,
        outcome="passed" if passed else "failed",
        assertion=assertion,
        expected=expected,
        actual=actual,
        detail="Regression assertion passed." if passed else "Regression assertion failed.",
    )


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
