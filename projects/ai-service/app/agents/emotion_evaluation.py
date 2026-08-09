"""情绪识别评测（agent_cases.json 中 expected.emotion 非空的用例）。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agents.emotion_classifier import RuleBasedEmotionClassifier
from app.agents.intent_evaluation import AgentEvalCase

_CLASSIFIER = RuleBasedEmotionClassifier()


class EmotionEvalCaseResult(BaseModel):
    case_id: str = Field(min_length=1)
    input_message: str
    expected: str
    actual: str
    passed: bool


class EmotionEvalSummary(BaseModel):
    case_count: int = Field(ge=0)
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    case_results: list[EmotionEvalCaseResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failed_case_count == 0


def evaluate_emotion_case(eval_case: AgentEvalCase) -> EmotionEvalCaseResult | None:
    """评估单个用例的情绪识别；无 expected.emotion 时返回 None（不属于情绪套件）。"""
    expected_emotion = eval_case.expected.emotion
    if not expected_emotion:
        return None
    actual_emotion = _CLASSIFIER.recognize(eval_case.inputs.message).emotion.value
    passed = actual_emotion == expected_emotion
    return EmotionEvalCaseResult(
        case_id=eval_case.id,
        input_message=eval_case.inputs.message,
        expected=expected_emotion,
        actual=actual_emotion,
        passed=passed,
    )


def evaluate_emotion_cases(cases: list[AgentEvalCase]) -> EmotionEvalSummary:
    results = [
        result
        for case in cases
        if (result := evaluate_emotion_case(case)) is not None
    ]
    passed_count = sum(1 for result in results if result.passed)
    return EmotionEvalSummary(
        case_count=len(results),
        passed_case_count=passed_count,
        failed_case_count=len(results) - passed_count,
        accuracy=passed_count / len(results) if results else 0.0,
        case_results=results,
    )


def format_emotion_eval_summary(summary: EmotionEvalSummary) -> list[str]:
    lines = [
        f"emotion accuracy={summary.accuracy:.3f} "
        f"({summary.passed_case_count}/{summary.case_count})",
    ]
    for result in summary.case_results:
        if not result.passed:
            lines.append(
                f"  FAIL {result.case_id}: expected={result.expected} actual={result.actual}"
            )
    return lines


def format_emotion_bad_cases(summary: EmotionEvalSummary) -> list[str]:
    return format_emotion_eval_summary(summary)
