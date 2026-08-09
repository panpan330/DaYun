from app.agents.multiturn_evaluation import evaluate_multiturn_cases
from app.schemas.multiturn_eval import MultiturnEvalCase


class _FakeClassification:
    def __init__(self, intent: str, route: str) -> None:
        self.intent = intent
        self.intent_route = route


def _classifier_with_history() -> object:
    """Rule-based fake: '那我要退款' 无历史→neutral，有历史→refund_request；其余直判。"""

    def classify(message: str, *, history: list[str] | None = None) -> _FakeClassification:
        if "退款" in message and not history:
            return _FakeClassification("neutral", "unsupported")
        if "退款" in message:
            return _FakeClassification("refund_request", "refund_order")
        if "发货" in message or "订单" in message:
            return _FakeClassification("order_query", "query_order")
        return _FakeClassification("neutral", "unsupported")

    return classify


def _case() -> MultiturnEvalCase:
    return MultiturnEvalCase(
        id="mt_order_refund_001",
        name="跨轮：查订单后问退款",
        conversation=[
            {
                "message": "我的订单 202501010001 发货了吗",
                "expected_intent": "order_query",
                "expected_route": "query_order",
                "requires_history": False,
            },
            {
                "message": "那我要退款",
                "expected_intent": "refund_request",
                "expected_route": "refund_order",
                "requires_history": True,
            },
        ],
        final_answer={"expected_intent": "refund_request"},
    )


def test_multiturn_all_pass() -> None:
    summary = evaluate_multiturn_cases([_case()], classifier=_classifier_with_history())
    assert summary.case_count == 1
    assert summary.passed_case_count == 1
    assert summary.failed_case_count == 0


def test_multiturn_requires_history_is_enforced_in_strict_mode() -> None:
    # 无历史时也判对 → requires_history 断言过松 → 该轮 fail（strict_history=True 时）
    def loose_classify(message: str, *, history: list[str] | None = None) -> _FakeClassification:
        if "退款" in message:
            return _FakeClassification("refund_request", "refund_order")
        return _FakeClassification("order_query", "query_order")

    summary = evaluate_multiturn_cases([_case()], classifier=loose_classify, strict_history=True)
    assert summary.passed_case_count == 0
    assert summary.failed_case_count == 1
    assert any("requires_history" in result.reason for result in summary.bad_cases[0].turn_results)


def test_multiturn_fails_when_context_intent_wrong() -> None:
    def wrong_classify(message: str, *, history: list[str] | None = None) -> _FakeClassification:
        return _FakeClassification("neutral", "unsupported")

    summary = evaluate_multiturn_cases([_case()], classifier=wrong_classify)
    assert summary.passed_case_count == 0
    assert summary.bad_cases[0].case_id == "mt_order_refund_001"


def test_multiturn_final_answer_asserted() -> None:
    def no_final(message: str, *, history: list[str] | None = None) -> _FakeClassification:
        if "退款" in message and history:
            return _FakeClassification("order_query", "query_order")
        if "退款" in message:
            return _FakeClassification("refund_request", "refund_order")
        return _FakeClassification("order_query", "query_order")

    summary = evaluate_multiturn_cases([_case()], classifier=no_final)
    assert summary.passed_case_count == 0
    assert summary.bad_cases[0].final_answer_passed is False


def test_multiturn_non_strict_mode_passes_when_context_matches() -> None:
    # 规则模式（strict_history=False）：requires_history 轮只要 context 判对即可
    def loose_classify(message: str, *, history: list[str] | None = None) -> _FakeClassification:
        if "退款" in message:
            return _FakeClassification("refund_request", "refund_order")
        return _FakeClassification("order_query", "query_order")

    summary = evaluate_multiturn_cases([_case()], classifier=loose_classify, strict_history=False)
    assert summary.passed_case_count == 1


def test_multiturn_uses_rule_based_default_classifier() -> None:
    # 默认 classifier（RuleBasedTicketIntentClassifier）也应跑通结构，不抛错
    summary = evaluate_multiturn_cases([_case()])
    assert summary.case_count == 1
    assert isinstance(summary.passed_case_count, int)
