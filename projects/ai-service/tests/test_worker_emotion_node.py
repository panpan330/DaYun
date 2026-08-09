from app.agents.emotion_classifier import CustomerEmotion, RuleBasedEmotionClassifier
from app.agents.workers.emotion_node import make_emotion_recognition_node, route_after_emotion

ANGRY_MESSAGE = "气死我了！我的订单 202501010001 还不发货！我要投诉！"
NEUTRAL_MESSAGE = "你好，我想查一下订单状态"


def test_emotion_node_short_circuits_on_handoff_emotion():
    node = make_emotion_recognition_node(RuleBasedEmotionClassifier())
    result = node({"normalized_message": ANGRY_MESSAGE, "user_message": ANGRY_MESSAGE})
    assert result.get("emotion_handoff_requested") is True
    assert "final_answer" in result
    assert "人工客服" in result["final_answer"]


def test_emotion_node_passes_through_when_neutral():
    node = make_emotion_recognition_node(RuleBasedEmotionClassifier())
    result = node({"normalized_message": NEUTRAL_MESSAGE, "user_message": NEUTRAL_MESSAGE})
    assert result.get("emotion_handoff_requested") is not True
    assert "final_answer" not in result
    assert result.get("emotion") == CustomerEmotion.NEUTRAL


def test_route_after_emotion():
    assert route_after_emotion({"emotion_handoff_requested": True}) == "handoff"
    assert route_after_emotion({"emotion_handoff_requested": False}) == "continue"
    assert route_after_emotion({}) == "continue"


def test_normalize_user_input_resets_emotion_fields():
    """带 checkpointer 时上一轮 emotion 字段不得跨轮残留（每轮独立识别）。"""
    from app.agents.ticket_agent import normalize_user_input_node

    result = normalize_user_input_node(
        {
            "user_message": "你好",
            "emotion": "angry",
            "emotion_reason": "上一轮愤怒",
            "emotion_handoff_requested": True,
        }
    )
    assert result.get("emotion_handoff_requested") is False
    assert result.get("emotion") is None
    assert result.get("emotion_reason") is None
    assert result.get("normalized_message") == "你好"
