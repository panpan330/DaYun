"""图集成测试：情绪识别节点、转人工分支、跳过写工具。"""

from app.agents.emotion_classifier import (
    CustomerEmotion,
    FakeLLMEmotionClassifier,
    RuleBasedEmotionClassifier,
)
from app.agents.ticket_agent import build_ticket_agent_graph_for_model_mode


def _run_emotion_case(message: str, mode: str = "rule_based"):
    graph = build_ticket_agent_graph_for_model_mode(
        mode=mode,
        emotion_classifier=(
            RuleBasedEmotionClassifier() if mode == "rule_based" else FakeLLMEmotionClassifier()
        ),
    )
    result = graph.invoke(
        {
            "user_message": message,
            "agent_trace_id": "trace-emotion-test",
            "node_history": [],
        }
    )
    return result


def test_angry_message_goes_handoff_and_skips_tools():
    result = _run_emotion_case("你们太气人了，这是骗子！我要投诉退款！")
    assert result["emotion"] == CustomerEmotion.ANGRY
    assert result["emotion_handoff_requested"] is True
    # 写操作未被触发
    assert result.get("created_ticket") is None
    assert result.get("refund_status") is None
    assert result.get("cancel_status") is None
    assert "人工" in result["final_answer"]


def test_neutral_message_continues_normal_flow():
    result = _run_emotion_case("退款多久到账？")
    assert result["emotion"] == CustomerEmotion.NEUTRAL
    assert result["emotion_handoff_requested"] is False
    # 正常流程走 policy 回答
    assert result["intent"] == "policy_question"


def test_dissatisfied_does_not_trigger_handoff():
    result = _run_emotion_case("我对这个处理结果很不满意，很失望")
    assert result["emotion"] == CustomerEmotion.DISSATISFIED
    assert result["emotion_handoff_requested"] is False


def test_emotion_handoff_has_reason():
    result = _run_emotion_case("请快点处理，立刻马上！")
    assert result["emotion_handoff_requested"] is True
    assert result.get("emotion_reason")
