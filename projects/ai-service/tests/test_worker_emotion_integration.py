from app.agents.emotion_classifier import RuleBasedEmotionClassifier
from app.agents.workers.knowledge_agent import build_knowledge_agent_graph
from app.agents.workers.order_agent import build_order_agent_graph
from app.agents.workers.ticket_worker import build_ticket_worker_graph
from tests.tool_fakes import FakePolicyRagService, FakeTicketCreator, make_policy_rag_answer

ANGRY = "气死我了！我的订单 202501010001 还不发货！我要投诉！"


def test_order_worker_short_circuits_on_angry():
    graph = build_order_agent_graph(
        emotion_classifier=RuleBasedEmotionClassifier(),
    )
    result = graph.invoke(
        {
            "normalized_message": ANGRY,
            "user_message": ANGRY,
            "agent_trace_id": "t-1",
        }
    )
    assert result.get("emotion_handoff_requested") is True
    assert "final_answer" in result
    assert "人工客服" in result["final_answer"]
    # 原业务节点未执行
    assert "order_query_status" not in result


def test_knowledge_worker_short_circuits_on_angry():
    graph = build_knowledge_agent_graph(
        service=FakePolicyRagService(
            make_policy_rag_answer(answer="退货政策是 30 天无理由。")
        ),
        emotion_classifier=RuleBasedEmotionClassifier(),
    )
    result = graph.invoke(
        {
            "normalized_message": ANGRY,
            "user_message": ANGRY,
            "agent_trace_id": "t-1",
        }
    )
    assert result.get("emotion_handoff_requested") is True
    assert "final_answer" in result
    assert "rag_answer_status" not in result


def test_ticket_worker_short_circuits_on_angry():
    graph = build_ticket_worker_graph(
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=False,
        emotion_classifier=RuleBasedEmotionClassifier(),
    )
    result = graph.invoke(
        {
            "normalized_message": ANGRY,
            "user_message": ANGRY,
            "agent_trace_id": "t-1",
        }
    )
    assert result.get("emotion_handoff_requested") is True
    assert "final_answer" in result


def test_neutral_message_runs_original_flow():
    graph = build_order_agent_graph(
        emotion_classifier=RuleBasedEmotionClassifier(),
    )
    result = graph.invoke(
        {
            "normalized_message": "你好，查一下订单状态",
            "user_message": "你好，查一下订单状态",
            "agent_trace_id": "t-1",
        }
    )
    assert result.get("emotion_handoff_requested") is not True
    # 原流程执行：缺订单号 → missing_order_id
    assert result.get("order_query_status") == "missing_order_id"
