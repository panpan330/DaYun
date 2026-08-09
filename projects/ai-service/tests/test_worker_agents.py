from app.agents.workers.knowledge_agent import build_knowledge_agent_graph
from app.agents.workers.order_agent import build_order_agent_graph
from app.agents.workers.ticket_worker import build_ticket_worker_graph
from tests.tool_fakes import (
    FakeNoContextPolicyRagService,
    FakePolicyRagService,
    FakeTicketCreator,
    make_policy_rag_answer,
)
from app.schemas.tool import QueryOrderArgs, QueryOrderResult


def test_knowledge_agent_answers_policy_question() -> None:
    graph = build_knowledge_agent_graph(
        service=FakePolicyRagService(
            make_policy_rag_answer(answer="退货政策是 30 天无理由。")
        )
    )
    result = graph.invoke({"normalized_message": "退货政策是什么"})
    assert result["rag_answer_status"] == "answered"
    assert "退货政策" in (result.get("final_answer") or "")


def test_knowledge_agent_marks_no_context_for_ticket_transfer() -> None:
    graph = build_knowledge_agent_graph(
        service=FakeNoContextPolicyRagService()
    )
    result = graph.invoke({"normalized_message": "完全不知道的问题"})
    assert result["rag_answer_status"] == "no_context"


def test_knowledge_agent_marks_ticket_need_for_no_context_transfer() -> None:
    graph = build_knowledge_agent_graph(
        service=FakeNoContextPolicyRagService()
    )
    result = graph.invoke(
        {
            "normalized_message": "完全不知道的问题",
            "intent": "policy_question",
        }
    )
    assert result["needs_ticket"] is True
    assert result["ticket_need_source"] == "rag_no_context"


def test_order_agent_queries_order() -> None:
    def executor(arguments: QueryOrderArgs) -> QueryOrderResult:
        return QueryOrderResult(
            order_id=arguments.order_id,
            order_status="shipped",
            payment_status="paid",
            logistics_message="已发货",
            latest_event="包裹已发出",
            can_create_ticket=True,
            source="java_business_service",
        )

    graph = build_order_agent_graph(order_query_executor=executor)
    result = graph.invoke({"normalized_message": "查订单 A1001 物流"})
    assert result["order_query_status"] == "succeeded"
    assert result["order_query_result"]["order_id"] == "A1001"


def test_order_agent_missing_order_id() -> None:
    graph = build_order_agent_graph()
    result = graph.invoke({"normalized_message": "查一下订单状态"})
    assert result["order_query_status"] == "missing_order_id"
    assert result["order_query_error_code"] is not None


def test_ticket_worker_creates_ticket_after_confirmation() -> None:
    graph = build_ticket_worker_graph(
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "normalized_message": "申请退款，订单 A1001 破损",
            "ticket_confirmation_approved": True,
            "ticket_fields": {
                "issue_type": "refund",
                "order_id": "A1001",
                "description": "订单破损",
                "user_request": "申请退款",
                "urgency": "high",
                "need_human_review": False,
            },
        }
    )
    assert result["ticket_creation_status"] == "created"
    assert result["created_ticket"]["ticket_id"] is not None
