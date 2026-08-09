"""情绪评测测试：emotion 套件（只评 expected.emotion 非空用例）。"""

from app.agents.emotion_evaluation import (
    EmotionEvalCaseResult,
    evaluate_emotion_case,
    evaluate_emotion_cases,
    format_emotion_eval_summary,
)
from app.agents.intent_evaluation import (
    AgentEvalCase,
    AgentEvalExpected,
    AgentEvalInputs,
    AgentEvalMetadata,
)


def _make_case(
    case_id: str,
    message: str,
    emotion: str | None,
) -> AgentEvalCase:
    return AgentEvalCase(
        id=case_id,
        name=f"情绪用例-{case_id}",
        inputs=AgentEvalInputs(message=message, history=[]),
        expected=AgentEvalExpected(
            intent="ticket_request",
            intent_route="decide_ticket_need",
            emotion=emotion,
        ),
        metadata=AgentEvalMetadata(
            task_type="emotion",
            business_domain="general",
            case_type="normal",
            difficulty="easy",
            priority="p0",
            tags=["emotion", "regression"],
        ),
    )


def test_evaluate_emotion_case_hit():
    case = _make_case("emotion_angry_001", "你们太气人了，我要投诉退款！", "angry")
    result = evaluate_emotion_case(case)
    assert isinstance(result, EmotionEvalCaseResult)
    assert result.passed is True
    assert result.expected == "angry"
    assert result.actual == "angry"


def test_evaluate_emotion_case_miss():
    case = _make_case("emotion_neutral_001", "退款多久到账？", "angry")
    result = evaluate_emotion_case(case)
    assert result.passed is False
    assert result.actual == "neutral"


def test_evaluate_emotion_cases_skips_cases_without_emotion():
    cases = [
        _make_case("c1", "气死我了", "angry"),
        _make_case("c2", "谢谢", "satisfied"),
        _make_case("c3", "退款多久到账？", None),  # 无 emotion 字段 → 跳过
    ]
    summary = evaluate_emotion_cases(cases)
    assert summary.case_count == 2
    assert summary.passed_case_count == 2
    assert summary.failed_case_count == 0


def test_evaluate_emotion_cases_reports_failures():
    cases = [
        _make_case("c1", "气死我了", "angry"),
        _make_case("c2", "谢谢", "angry"),
    ]
    summary = evaluate_emotion_cases(cases)
    assert summary.passed_case_count == 1
    assert summary.failed_case_count == 1
    assert summary.accuracy == 0.5
    assert summary.passed is False


def test_format_summary_contains_accuracy():
    cases = [_make_case("c1", "气死我了", "angry")]
    summary = evaluate_emotion_cases(cases)
    lines = format_emotion_eval_summary(summary)
    assert any("accuracy" in line for line in lines)
