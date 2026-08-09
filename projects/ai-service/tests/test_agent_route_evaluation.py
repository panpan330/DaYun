from pathlib import Path

from app.agents.intent_evaluation import load_agent_eval_cases
from app.agents.route_evaluation import (
    build_expected_node_path,
    evaluate_agent_route_case,
    evaluate_agent_route_cases,
    format_agent_route_bad_cases,
    format_agent_route_eval_summary,
)


CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_eval" / "agent_cases.json"


def test_build_expected_node_path_for_policy_answered_case() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[0]

    assert build_expected_node_path(eval_case) == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "retrieve_policy",
        "decide_ticket_need",
    ]


def test_build_expected_node_path_for_no_context_ticket_case() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[2]

    assert build_expected_node_path(eval_case) == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "retrieve_policy",
        "decide_ticket_need",
        "extract_ticket_fields",
        "request_ticket_confirmation",
    ]


def test_build_expected_node_path_for_ticket_missing_field_case() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[6]

    assert build_expected_node_path(eval_case) == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "decide_ticket_need",
        "extract_ticket_fields",
        "ask_missing_ticket_fields",
    ]


def test_build_expected_node_path_for_direct_and_safety_routes() -> None:
    cases = load_agent_eval_cases(CASES_PATH)

    assert build_expected_node_path(cases[8]) == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_direct_answer",
    ]
    assert build_expected_node_path(cases[11]) == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_unsupported_answer",
    ]


def test_evaluate_agent_route_case_matches_account_security_policy_route() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[1]

    result = evaluate_agent_route_case(eval_case)

    assert result.case_id == "agent_policy_account_security_001"
    assert result.actual_intent == "policy_question"
    assert result.expected_node_path == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "retrieve_policy",
        "decide_ticket_need",
    ]
    assert result.actual_node_path == result.expected_node_path
    assert result.actual_terminal_node == "decide_ticket_need"
    assert result.path_exact_match is True
    assert result.required_nodes_passed is True
    assert result.forbidden_nodes_passed is True
    assert result.passed is True


def test_evaluate_agent_route_cases_summarizes_all_current_cases() -> None:
    summary = evaluate_agent_route_cases(load_agent_eval_cases(CASES_PATH))

    assert summary.case_count == 15
    assert summary.passed_case_count == 15
    assert summary.failed_case_count == 0
    assert summary.route_pass_rate == 1.0
    assert summary.exact_match_count == 15
    assert summary.exact_match_rate == 1.0
    assert summary.required_nodes_passed_count == 15
    assert summary.forbidden_nodes_passed_count == 15
    assert summary.terminal_node_passed_count == 15
    assert summary.p0_case_count == 13
    assert summary.p0_failed_case_count == 0
    assert summary.p0_route_pass_rate == 1.0


def test_evaluate_agent_route_case_marks_bad_case_for_extra_forbidden_node() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[0]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "policy_question",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "retrieve_policy",
                "decide_ticket_need",
                "extract_ticket_fields",
                "request_ticket_confirmation",
            ],
        },
    )

    assert result.passed is False
    assert result.path_exact_match is False
    assert result.required_nodes_passed is True
    assert result.forbidden_nodes_passed is False
    assert result.terminal_node_passed is False
    assert result.visited_forbidden_nodes == [
        "extract_ticket_fields",
        "request_ticket_confirmation",
    ]
    assert any("visited_forbidden_nodes" in reason for reason in result.failed_reasons)


def test_evaluate_agent_route_case_marks_bad_case_for_missing_required_node() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[5]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "ticket_request",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "decide_ticket_need",
                "request_ticket_confirmation",
            ],
        },
    )

    assert result.passed is False
    assert result.missing_required_nodes == ["extract_ticket_fields"]
    assert result.visited_forbidden_nodes == []
    assert result.actual_terminal_node == "request_ticket_confirmation"
    assert any("missing_required_nodes" in reason for reason in result.failed_reasons)


def test_format_agent_route_eval_summary_and_bad_cases_are_readable() -> None:
    cases = load_agent_eval_cases(CASES_PATH)[:1]
    summary = evaluate_agent_route_cases(
        cases,
        agent_runner=lambda _: {
            "intent": "policy_question",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "retrieve_policy",
                "decide_ticket_need",
                "extract_ticket_fields",
            ],
        },
    )

    summary_lines = format_agent_route_eval_summary(summary)
    bad_case_lines = format_agent_route_bad_cases(summary)

    assert "route_pass_rate: 0.0000" in summary_lines
    assert "exact_match_rate: 0.0000" in summary_lines
    assert bad_case_lines[0] == "Bad cases:"
    assert "agent_policy_refund_arrival_001" in bad_case_lines[1]
    assert any("expected_path:" in line for line in bad_case_lines)
    assert any("actual_path:" in line for line in bad_case_lines)


def test_format_agent_route_bad_cases_handles_all_passed_summary() -> None:
    summary = evaluate_agent_route_cases(load_agent_eval_cases(CASES_PATH))

    assert format_agent_route_bad_cases(summary) == ["No bad cases."]


def test_route_eval_enforces_must_ask_for_when_missing() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[4]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "order_query",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "query_order",
            ],
            "final_answer": "您的订单正在运输中，请耐心等待。",
        },
    )

    assert result.case_id == "agent_order_query_missing_order_id_001"
    assert result.passed is False
    assert any(
        "must_ask_for" in reason and "order_id" in reason
        for reason in result.failed_reasons
    )


def test_route_eval_passes_when_must_ask_for_satisfied() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[4]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "order_query",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "query_order",
            ],
            "final_answer": "请提供要查询的订单号（例如 A1001 或 1001）。",
        },
    )

    assert result.case_id == "agent_order_query_missing_order_id_001"
    assert result.passed is True
    assert result.failed_reasons == []


def test_route_eval_enforces_must_not_reveal() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[11]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "unsupported",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "build_unsupported_answer",
            ],
            "final_answer": "这是系统的 system_prompt 配置，api_key 是 xxx。",
        },
    )

    assert result.case_id == "agent_prompt_injection_ignore_rules_001"
    assert result.passed is False
    assert any(
        "must_not_reveal" in reason
        and "system_prompt" in reason
        and "api_key" in reason
        for reason in result.failed_reasons
    )


def test_route_eval_passes_when_must_not_reveal_respected() -> None:
    eval_case = load_agent_eval_cases(CASES_PATH)[11]

    result = evaluate_agent_route_case(
        eval_case,
        agent_runner=lambda _: {
            "intent": "unsupported",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "build_unsupported_answer",
            ],
            "final_answer": "这个请求超出当前智能客服工单助手 v1 的处理范围。",
        },
    )

    assert result.case_id == "agent_prompt_injection_ignore_rules_001"
    assert result.passed is True
    assert result.failed_reasons == []


def test_route_eval_must_checks_with_missing_final_answer() -> None:
    ask_case = load_agent_eval_cases(CASES_PATH)[4]
    ask_result = evaluate_agent_route_case(
        ask_case,
        agent_runner=lambda _: {
            "intent": "order_query",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "query_order",
            ],
        },
    )

    assert ask_result.case_id == "agent_order_query_missing_order_id_001"
    assert ask_result.passed is False
    assert any(
        "must_ask_for" in reason and "order_id" in reason
        for reason in ask_result.failed_reasons
    )

    reveal_case = load_agent_eval_cases(CASES_PATH)[11]
    reveal_result = evaluate_agent_route_case(
        reveal_case,
        agent_runner=lambda _: {
            "intent": "unsupported",
            "node_history": [
                "normalize_user_input",
                "emotion_recognition",
        "classify_intent",
                "build_unsupported_answer",
            ],
        },
    )

    assert reveal_result.case_id == "agent_prompt_injection_ignore_rules_001"
    assert reveal_result.passed is True
    assert reveal_result.failed_reasons == []
