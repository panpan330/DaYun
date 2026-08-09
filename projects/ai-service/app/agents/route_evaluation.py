from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.agents.emotion_classifier import EMOTION_HANDOFF_EMOTIONS
from app.agents.intent_evaluation import AgentEvalCase, AgentEvalPriority
from app.agents.must_check import check_must_ask_for, check_must_not_reveal
from app.agents.ticket_agent import run_ticket_agent


AgentRunner = Callable[[str], Mapping[str, Any]]

COMMON_ENTRY_NODES = ["normalize_user_input", "emotion_recognition", "classify_intent"]
EMOTION_HANDOFF_NODE_PATH = ["normalize_user_input", "emotion_recognition", "build_emotion_handoff_answer"]
ALL_AGENT_ROUTE_NODES = {
    "normalize_user_input",
    "classify_intent",
    "retrieve_policy",
    "decide_ticket_need",
    "query_order",
    "extract_ticket_fields",
    "ask_missing_ticket_fields",
    "request_ticket_confirmation",
    "create_ticket",
    "handle_refund_request",
    "execute_refund_request",
    "build_direct_answer",
    "build_unsupported_answer",
    "ask_clarifying_question",
}


class AgentRouteEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    expected_intent: str = Field(min_length=1)
    actual_intent: str | None = None
    expected_node_path: list[str] = Field(default_factory=list)
    actual_node_path: list[str] = Field(default_factory=list)
    expected_terminal_node: str | None = None
    actual_terminal_node: str | None = None
    required_nodes: list[str] = Field(default_factory=list)
    missing_required_nodes: list[str] = Field(default_factory=list)
    forbidden_nodes: list[str] = Field(default_factory=list)
    visited_forbidden_nodes: list[str] = Field(default_factory=list)
    unexpected_nodes: list[str] = Field(default_factory=list)
    path_exact_match: bool
    required_nodes_passed: bool
    forbidden_nodes_passed: bool
    terminal_node_passed: bool
    priority: AgentEvalPriority
    task_type: str = Field(min_length=1)
    business_domain: str = Field(min_length=1)
    case_type: str = Field(min_length=1)
    passed: bool
    failed_reasons: list[str] = Field(default_factory=list)


class AgentRouteEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    route_pass_rate: float = Field(ge=0, le=1)
    exact_match_count: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0, le=1)
    required_nodes_passed_count: int = Field(ge=0)
    forbidden_nodes_passed_count: int = Field(ge=0)
    terminal_node_passed_count: int = Field(ge=0)
    p0_case_count: int = Field(ge=0)
    p0_passed_case_count: int = Field(ge=0)
    p0_failed_case_count: int = Field(ge=0)
    p0_route_pass_rate: float = Field(ge=0, le=1)
    results: list[AgentRouteEvalCaseResult] = Field(default_factory=list)


def build_expected_node_path(eval_case: AgentEvalCase) -> list[str]:
    intent = eval_case.expected.intent
    expected_ticket = _expected_ticket(eval_case)

    expected_emotion = eval_case.expected.emotion
    if expected_emotion and expected_emotion in {
        emotion.value for emotion in EMOTION_HANDOFF_EMOTIONS
    }:
        # 触发情绪转人工的用例：期望路径是情绪识别 → 转人工分支
        return list(EMOTION_HANDOFF_NODE_PATH)

    if intent == "policy_question":
        path = COMMON_ENTRY_NODES + ["retrieve_policy", "decide_ticket_need"]
        if expected_ticket.get("should_create_ticket") is True:
            path += _expected_ticket_tail(expected_ticket)
        return path

    if intent == "order_query":
        return COMMON_ENTRY_NODES + ["query_order"]

    if intent == "ticket_request":
        path = COMMON_ENTRY_NODES + ["decide_ticket_need"]
        path += _expected_ticket_tail(expected_ticket)
        return path

    if intent == "refund_request":
        path = COMMON_ENTRY_NODES + ["handle_refund_request"]
        path += _expected_refund_tail(expected_ticket)
        return path

    if intent == "smalltalk":
        return COMMON_ENTRY_NODES + ["build_direct_answer"]

    if intent == "unsupported":
        return COMMON_ENTRY_NODES + ["build_unsupported_answer"]

    if intent == "unclear":
        return COMMON_ENTRY_NODES + ["ask_clarifying_question"]

    raise ValueError(f"unsupported intent for route eval: {intent!r}")


def evaluate_agent_route_case(
    eval_case: AgentEvalCase,
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> AgentRouteEvalCaseResult:
    actual_state = dict(agent_runner(eval_case.inputs.message))
    expected_node_path = build_expected_node_path(eval_case)
    actual_node_path = _actual_node_path(actual_state)
    required_nodes = list(expected_node_path)
    forbidden_nodes = sorted(ALL_AGENT_ROUTE_NODES - set(expected_node_path))
    missing_required_nodes = [
        node for node in required_nodes if node not in actual_node_path
    ]
    visited_forbidden_nodes = [
        node for node in actual_node_path if node in forbidden_nodes
    ]
    unexpected_nodes = [
        node for node in actual_node_path if node not in expected_node_path
    ]
    expected_terminal_node = _terminal_node(expected_node_path)
    actual_terminal_node = _terminal_node(actual_node_path)
    path_exact_match = actual_node_path == expected_node_path
    required_nodes_passed = not missing_required_nodes
    forbidden_nodes_passed = not visited_forbidden_nodes
    terminal_node_passed = actual_terminal_node == expected_terminal_node
    failed_reasons = _collect_failed_reasons(
        expected_node_path=expected_node_path,
        actual_node_path=actual_node_path,
        missing_required_nodes=missing_required_nodes,
        visited_forbidden_nodes=visited_forbidden_nodes,
        expected_terminal_node=expected_terminal_node,
        actual_terminal_node=actual_terminal_node,
    )
    failed_reasons += _collect_must_check_failed_reasons(
        eval_case=eval_case,
        actual_state=actual_state,
    )

    return AgentRouteEvalCaseResult(
        case_id=eval_case.id,
        name=eval_case.name,
        message=eval_case.inputs.message,
        expected_intent=eval_case.expected.intent,
        actual_intent=_optional_string(actual_state.get("intent")),
        expected_node_path=expected_node_path,
        actual_node_path=actual_node_path,
        expected_terminal_node=expected_terminal_node,
        actual_terminal_node=actual_terminal_node,
        required_nodes=required_nodes,
        missing_required_nodes=missing_required_nodes,
        forbidden_nodes=forbidden_nodes,
        visited_forbidden_nodes=visited_forbidden_nodes,
        unexpected_nodes=unexpected_nodes,
        path_exact_match=path_exact_match,
        required_nodes_passed=required_nodes_passed,
        forbidden_nodes_passed=forbidden_nodes_passed,
        terminal_node_passed=terminal_node_passed,
        priority=eval_case.metadata.priority,
        task_type=eval_case.metadata.task_type,
        business_domain=eval_case.metadata.business_domain,
        case_type=eval_case.metadata.case_type,
        passed=not failed_reasons,
        failed_reasons=failed_reasons,
    )


def evaluate_agent_route_cases(
    cases: Sequence[AgentEvalCase],
    *,
    agent_runner: AgentRunner = run_ticket_agent,
) -> AgentRouteEvalSummary:
    results = [
        evaluate_agent_route_case(eval_case, agent_runner=agent_runner)
        for eval_case in cases
    ]
    p0_results = [result for result in results if result.priority == "p0"]

    return AgentRouteEvalSummary(
        case_count=len(results),
        passed_case_count=sum(1 for result in results if result.passed),
        failed_case_count=sum(1 for result in results if not result.passed),
        route_pass_rate=_ratio(
            sum(1 for result in results if result.passed),
            len(results),
        ),
        exact_match_count=sum(1 for result in results if result.path_exact_match),
        exact_match_rate=_ratio(
            sum(1 for result in results if result.path_exact_match),
            len(results),
        ),
        required_nodes_passed_count=sum(
            1 for result in results if result.required_nodes_passed
        ),
        forbidden_nodes_passed_count=sum(
            1 for result in results if result.forbidden_nodes_passed
        ),
        terminal_node_passed_count=sum(
            1 for result in results if result.terminal_node_passed
        ),
        p0_case_count=len(p0_results),
        p0_passed_case_count=sum(1 for result in p0_results if result.passed),
        p0_failed_case_count=sum(1 for result in p0_results if not result.passed),
        p0_route_pass_rate=_ratio(
            sum(1 for result in p0_results if result.passed),
            len(p0_results),
        ),
        results=results,
    )


def format_agent_route_eval_summary(summary: AgentRouteEvalSummary) -> list[str]:
    return [
        "Agent route evaluation summary",
        f"cases: {summary.case_count}",
        f"passed_cases: {summary.passed_case_count}",
        f"failed_cases: {summary.failed_case_count}",
        f"route_pass_rate: {summary.route_pass_rate:.4f}",
        f"exact_match_count: {summary.exact_match_count}",
        f"exact_match_rate: {summary.exact_match_rate:.4f}",
        f"required_nodes_passed_count: {summary.required_nodes_passed_count}",
        f"forbidden_nodes_passed_count: {summary.forbidden_nodes_passed_count}",
        f"terminal_node_passed_count: {summary.terminal_node_passed_count}",
        f"p0_cases: {summary.p0_case_count}",
        f"p0_passed_cases: {summary.p0_passed_case_count}",
        f"p0_failed_cases: {summary.p0_failed_case_count}",
        f"p0_route_pass_rate: {summary.p0_route_pass_rate:.4f}",
    ]


def format_agent_route_bad_cases(summary: AgentRouteEvalSummary) -> list[str]:
    bad_cases = [result for result in summary.results if not result.passed]
    if not bad_cases:
        return ["No bad cases."]

    lines = ["Bad cases:"]
    for result in bad_cases:
        lines.append(
            f"- {result.case_id}: priority={result.priority} "
            f"task_type={result.task_type} terminal={result.actual_terminal_node or '-'}"
        )
        lines.append(f"  expected_path: {' -> '.join(result.expected_node_path)}")
        lines.append(f"  actual_path: {' -> '.join(result.actual_node_path)}")
        for reason in result.failed_reasons:
            lines.append(f"  - {reason}")
    return lines


def _expected_ticket(eval_case: AgentEvalCase) -> dict[str, Any]:
    ticket = eval_case.expected.ticket
    if not isinstance(ticket, dict):
        return {}
    return ticket


def _expected_ticket_tail(expected_ticket: Mapping[str, Any]) -> list[str]:
    missing_fields = _string_list(expected_ticket.get("missing_ticket_fields"))
    if missing_fields:
        return ["extract_ticket_fields", "ask_missing_ticket_fields"]
    if expected_ticket.get("confirmation_required") is True:
        return ["extract_ticket_fields", "request_ticket_confirmation"]
    return ["extract_ticket_fields"]


def _expected_refund_tail(expected_ticket: Mapping[str, Any]) -> list[str]:
    missing_fields = _string_list(expected_ticket.get("missing_ticket_fields"))
    if missing_fields:
        return ["ask_missing_ticket_fields"]
    if expected_ticket.get("confirmation_required") is True:
        return ["request_ticket_confirmation"]
    return []


def _actual_node_path(actual_state: Mapping[str, Any]) -> list[str]:
    node_history = actual_state.get("node_history")
    if not isinstance(node_history, Sequence) or isinstance(node_history, str):
        return []
    return [node for node in node_history if isinstance(node, str) and node.strip()]


def _collect_failed_reasons(
    *,
    expected_node_path: list[str],
    actual_node_path: list[str],
    missing_required_nodes: list[str],
    visited_forbidden_nodes: list[str],
    expected_terminal_node: str | None,
    actual_terminal_node: str | None,
) -> list[str]:
    reasons: list[str] = []
    if actual_node_path != expected_node_path:
        reasons.append(
            "node_path expected="
            f"{expected_node_path!r} actual={actual_node_path!r}"
        )
    if missing_required_nodes:
        reasons.append(f"missing_required_nodes={missing_required_nodes!r}")
    if visited_forbidden_nodes:
        reasons.append(f"visited_forbidden_nodes={visited_forbidden_nodes!r}")
    if actual_terminal_node != expected_terminal_node:
        reasons.append(
            "terminal_node expected="
            f"{expected_terminal_node!r} actual={actual_terminal_node!r}"
        )
    return reasons


def _collect_must_check_failed_reasons(
    *,
    eval_case: AgentEvalCase,
    actual_state: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if eval_case.expected.must_ask_for:
        reply = str(actual_state.get("final_answer") or "")
        missing = check_must_ask_for(reply, eval_case.expected.must_ask_for)
        if missing:
            reasons.append(f"must_ask_for: missing {', '.join(missing)}")
    if eval_case.expected.must_not_reveal:
        reply = str(actual_state.get("final_answer") or "")
        revealed = check_must_not_reveal(reply, eval_case.expected.must_not_reveal)
        if revealed:
            reasons.append(f"must_not_reveal: revealed {', '.join(revealed)}")
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


def _terminal_node(node_path: Sequence[str]) -> str | None:
    if not node_path:
        return None
    return node_path[-1]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
