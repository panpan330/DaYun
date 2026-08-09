"""Worker 入口情绪识别：命中强烈情绪时短路返回转人工回答。"""

from collections.abc import Callable

from app.agents.emotion_classifier import CustomerEmotionClassifier
from app.agents.ticket_agent import (
    RuleBasedEmotionClassifier,
    build_emotion_handoff_answer_node,
    emotion_recognition_node,
)


def make_emotion_recognition_node(
    classifier: CustomerEmotionClassifier | None = None,
) -> Callable[[dict], dict]:
    """Build a worker-entry emotion node that short-circuits on handoff emotions."""

    def node(state: dict) -> dict:
        emotion_result = emotion_recognition_node(
            state,
            classifier=classifier or RuleBasedEmotionClassifier(),
        )
        if emotion_result.get("emotion_handoff_requested"):
            answer = build_emotion_handoff_answer_node(emotion_result)
            return {**emotion_result, **answer}
        return emotion_result

    return node


def route_after_emotion(state: dict) -> str:
    """Conditional edge: short-circuit to END when handoff is requested."""
    if state.get("emotion_handoff_requested"):
        return "handoff"
    return "continue"
