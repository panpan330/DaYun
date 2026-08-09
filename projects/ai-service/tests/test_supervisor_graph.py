import pytest

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agents.supervisor.supervisor_graph import (
    SUPERVISOR_ROUTE_TABLE,
    build_supervisor_graph,
)
from app.agents.supervisor.supervisor_router import (
    FakeLLMSupervisorRouter,
    RuleSupervisorRouter,
    SupervisorRoute,
)
from app.core.exceptions import AppException
from tests.rag_fakes import make_retrieved_chunk
from tests.tool_fakes import (
    FakeCancelExecutor,
    FakeNoContextPolicyRagService,
    FakePolicyRagService,
    FakeRefundExecutor,
    FakeTicketCreator,
    make_created_ticket,
    make_policy_rag_answer,
)
from app.schemas.tool import QueryOrderArgs, QueryOrderResult


def _order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
    return QueryOrderResult(
        order_id=arguments.order_id,
        order_status="shipped",
        payment_status="paid",
        logistics_message="已发货",
        latest_event="包裹已发出",
        can_create_ticket=True,
        source="java_business_service",
    )


def _failing_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
    raise AppException(
        code="ORDER_SERVICE_UNKNOWN",
        message="order service temporarily unavailable",
        status_code=500,
    )


def test_route_table_maps_all_routes() -> None:
    for route in SupervisorRoute:
        assert route in SUPERVISOR_ROUTE_TABLE


def test_supervisor_routes_order_query_to_order_agent() -> None:
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.ORDER_QUERY),
        order_query_executor=_order_executor,
    )
    result = graph.invoke({"user_message": "查订单 A1001 物流"})
    assert result["intent"] == "order_query"
    assert "已发货" in (result.get("final_answer") or "")


def test_supervisor_routes_policy_to_knowledge_agent() -> None:
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.KNOWLEDGE_QUESTION),
        knowledge_service=FakePolicyRagService(
            make_policy_rag_answer(answer="退货政策是 30 天无理由。")
        ),
    )
    result = graph.invoke({"user_message": "退货政策是什么"})
    assert result["rag_answer_status"] == "answered"
    assert "退货政策" in (result.get("final_answer") or "")


def test_supervisor_routes_ticket_to_ticket_agent() -> None:
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "user_message": "申请退款，订单 A1001 破损",
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


def test_supervisor_smalltalk_builds_direct_answer() -> None:
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.SMALLTALK)
    )
    result = graph.invoke({"user_message": "你好"})
    assert result["final_answer"] is not None


def test_knowledge_no_context_transfers_to_ticket() -> None:
    """Knowledge 子图 RAG no_context 时，监督层应把流程转给 ticket worker 建工单。"""
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.KNOWLEDGE_QUESTION),
        knowledge_service=FakeNoContextPolicyRagService(),
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "user_message": "一个知识库没有的问题",
            "ticket_confirmation_approved": True,
            "ticket_fields": {
                "issue_type": "policy_gap",
                "order_id": None,
                "description": "一个知识库没有的问题",
                "user_request": "一个知识库没有的问题",
                "urgency": "medium",
                "need_human_review": True,
            },
        }
    )
    assert result["needs_ticket"] is True
    assert result["rag_answer_status"] == "no_context"
    assert result["ticket_need_source"] == "rag_no_context"
    # 转单语义完整验证：ticket worker 实际创建了 policy_gap 工单
    assert result["created_ticket"] is not None
    assert result["created_ticket"]["category"] == "policy_gap"


def test_supervisor_order_query_failure_writes_state_to_top_level() -> None:
    """order 子图查询失败时，order_query_* 状态必须写回监督层顶层。

    修复前 SupervisorState 无 order_query_* 键，子图输出被过滤，
    "订单失败转人工"行为（_human_handoff_from_state）在多 Agent 模式静默丢失。
    """
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.ORDER_QUERY),
        order_query_executor=_failing_order_executor,
    )
    result = graph.invoke({"user_message": "查订单 A1001"})
    assert result["intent"] == "order_query"
    assert result["order_query_status"] == "failed"
    assert result["order_query_error_code"] == "ORDER_SERVICE_UNKNOWN"
    assert result["order_query_error_action"] == "contact_human_support"
    assert result["order_query_order_id"] == "A1001"


def test_supervisor_ticket_actor_id_reaches_created_ticket() -> None:
    """ticket_actor_id 必须穿透监督层到达工单创建者。

    修复前 SupervisorState 无 ticket_actor_id 键，顶层输入被过滤，
    工单创建者退化为 DEFAULT_TICKET_ACTOR_ID。
    """
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "user_message": "申请退款，订单 A1001 破损",
            "ticket_actor_id": "user_agent_42",
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
    assert result["created_ticket"]["requester_id"] == "user_agent_42"


def test_ticket_interrupt_pause_and_resume() -> None:
    """interrupt_confirmation=True 时 ticket worker 应暂停并等待确认，resume 后完成建单。"""
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        interrupt_confirmation=True,
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "supervisor-interrupt-test"}}
    initial = graph.invoke(
        {"user_message": "申请退款，订单 A1001 破损"},
        config=config,
    )
    assert initial.get("__interrupt__")
    resumed = graph.invoke(
        Command(resume={"approved": True}),
        config=config,
    )
    assert resumed.get("ticket_creation_status") == "created"
    assert resumed.get("created_ticket") is not None


class _FailingLLMRouter:
    """模拟 LLM 模式 router 失败：route() 抛 AppException，
    route_with_fallback 捕获后回退规则路由（与 LLMSupervisorRouter 行为一致）。"""

    def route(self, message: str) -> SupervisorRoute:
        raise AppException(
            code="LLM_API_KEY_MISSING",
            message="LLM API key 未配置",
            status_code=500,
        )

    def route_with_fallback(self, message: str) -> tuple[SupervisorRoute, str]:
        try:
            return self.route(message), "llm"
        except AppException:
            return RuleSupervisorRouter().route(message), "rule_fallback"


def test_supervisor_llm_router_failure_falls_back_to_rule() -> None:
    """LLM 路由失败（LLM_API_KEY_MISSING）时监督图应回退规则路由并正常完成。"""
    graph = build_supervisor_graph(
        router=_FailingLLMRouter(),
        order_query_executor=_order_executor,
    )
    result = graph.invoke({"user_message": "查订单 A1001 物流"})
    assert result["intent"] == "order_query"
    assert "已发货" in (result.get("final_answer") or "")


def test_supervisor_routes_refund_request_executes_refund() -> None:
    """多 Agent 下 refund_request 必须走到 ticket_agent（ticket_worker 子图内
    退款链）并真正执行退款——与单 Agent 语义一致（规格 6.3 场景 A），
    而非退化为创建退款工单。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        refund_executor=FakeRefundExecutor(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "user_message": "我要退 A1002 的款",
            "ticket_confirmation_approved": True,
        }
    )
    assert result["intent"] == "refund_request"
    assert result["refund_status"] == "succeeded"
    assert "A1002" in (result.get("final_answer") or "")


def test_supervisor_refund_request_multi_turn_collects_order_id_then_executes() -> None:
    """多 Agent 多轮退款：第一轮缺订单号追问，第二轮补充后不得被 ORDER_KEYWORDS
    带到 order_agent，必须经 active-refund-collection 强制回退款链并执行退款。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        refund_executor=FakeRefundExecutor(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-refund-multi-turn-001"}}

    first = graph.invoke(
        {"user_message": "我要退款，商品破损"},
        config=config,
    )
    assert first["intent"] == "refund_request"
    assert first["missing_ticket_fields"] == ["order_id"]
    assert "订单号" in (first.get("final_answer") or "")

    second = graph.invoke(
        {"user_message": "订单号是 A1002", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["intent"] == "refund_request"
    assert second["refund_status"] == "succeeded"
    assert "A1002" in (second.get("final_answer") or "")


def test_supervisor_refund_pending_change_mind_to_ticket_creates_ticket_not_refund() -> None:
    """退款字段完整停在待确认后，用户改口建工单：确认必须建工单而非误执行退款。

    修复前残留 refund_request_active=True（handle_refund_request 写入、工单链
    不清除）使 route_by_ticket_confirmation 把工单确认误路由到
    execute_refund_request——真实执行退款，用工单 description 当退款原因。
    """
    refund_executor = FakeRefundExecutor()
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        refund_executor=refund_executor,
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-refund-changemind-001"}}

    first = graph.invoke(
        {"user_message": "我要退 A1002 的款，空调坏了"},
        config=config,
    )
    # 退款字段完整（订单号+原因），停在待确认，refund_request_active 残留 True
    assert first["intent"] == "refund_request"
    assert first["refund_request_active"] is True
    assert first["missing_ticket_fields"] == []

    # 改口：建投诉工单（含完整字段），确认后应建工单，不得执行退款
    second = graph.invoke(
        {
            "user_message": "我要投诉，订单号 A1002，商品破损，帮我建个工单",
            "ticket_confirmation_approved": True,
        },
        config=config,
    )
    assert second["intent"] == "ticket_request"
    assert second["ticket_creation_status"] == "created"
    assert second["created_ticket"]["related_order_id"] == "A1002"
    assert "商品破损" in (second["created_ticket"]["description"] or "")
    assert second["refund_request_active"] is False
    assert second.get("refund_status") is None
    assert refund_executor.calls == []
    assert "工单已创建" in (second.get("final_answer") or "")


def test_supervisor_routes_cancel_request_executes_cancel() -> None:
    """多 Agent 下 cancel_request 必须走到 ticket_agent（ticket_worker 子图内
    取消链）并真正执行取消——与单 Agent 语义一致，而非退化为创建取消工单。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        cancel_executor=FakeCancelExecutor(),
        interrupt_confirmation=False,
    )
    result = graph.invoke(
        {
            "user_message": "取消订单 A1002，不想要了",
            "ticket_confirmation_approved": True,
        }
    )
    assert result["intent"] == "cancel_request"
    assert result["cancel_status"] == "succeeded"
    assert "A1002" in (result.get("final_answer") or "")


def test_supervisor_cancel_request_multi_turn_collects_order_id_then_executes() -> None:
    """多 Agent 多轮取消：第一轮缺订单号追问，第二轮补充后不得被 ORDER_KEYWORDS
    带到 order_agent，必须经 active-cancel-collection 强制回取消链并执行取消。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        cancel_executor=FakeCancelExecutor(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-cancel-multi-turn-001"}}

    first = graph.invoke(
        {"user_message": "我要取消订单"},
        config=config,
    )
    assert first["intent"] == "cancel_request"
    assert first["missing_ticket_fields"] == ["order_id"]
    assert "订单号" in (first.get("final_answer") or "")

    second = graph.invoke(
        {"user_message": "订单号是 A1002，不想要了", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["intent"] == "cancel_request"
    assert second["cancel_status"] == "succeeded"
    assert "A1002" in (second.get("final_answer") or "")


def test_supervisor_multi_turn_rule_ticket_request_collects_missing_fields() -> None:
    """rule 路由下纯工单诉求（无退款动作短语）的多轮字段收集仍走工单创建链：
    新增的条件 START 边在 ticket_request 下必须落到 extract_ticket_fields，
    补充消息也不得被 ORDER_KEYWORDS 带偏到 order_agent。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-multi-turn-rule-ticket-001"}}

    first = graph.invoke(
        {"user_message": "我要投诉，商品破损"},
        config=config,
    )
    assert first["intent"] == "ticket_request"
    assert first["missing_ticket_fields"] == ["order_id"]

    second = graph.invoke(
        {"user_message": "订单号是 A1001", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["intent"] == "ticket_request"
    assert second["ticket_creation_status"] == "created"
    assert second["created_ticket"]["related_order_id"] == "A1001"
    # 第一轮提供的描述保留并随工单创建
    assert "商品破损" in (second["created_ticket"]["description"] or "")


def test_supervisor_multi_turn_rule_collects_missing_order_id() -> None:
    """多 Agent 多轮（rule 路由）：缺字段追问后，补充消息不得被 ORDER_KEYWORDS
    带偏到 order_agent，必须强制回退款链并合并第一轮字段执行退款。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        ticket_creator=FakeTicketCreator(),
        refund_executor=FakeRefundExecutor(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-multi-turn-rule-001"}}

    first = graph.invoke(
        {"user_message": "我要申请退款，商品破损"},
        config=config,
    )
    # 进入退款流程并触发缺订单号追问
    assert first["intent"] == "refund_request"
    assert first["refund_request_active"] is True
    assert first["missing_ticket_fields"] == ["order_id"]
    assert "订单号" in (first.get("final_answer") or "")

    # 第二轮补充订单号："订单号是 A1001" 在 rule 路由下会命中 ORDER_KEYWORDS，
    # 若没有 active-refund-collection 检查会被误判为 order_query
    second = graph.invoke(
        {"user_message": "订单号是 A1001", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["intent"] == "refund_request"
    assert second["missing_ticket_fields"] == []
    assert second["refund_status"] == "succeeded"
    assert second["refund_result"]["order_id"] == "A1001"
    # 第一轮提供的字段（商品破损）必须保留并合并进退款草稿的 description
    assert "商品破损" in (second["ticket_fields"]["description"] or "")


def test_supervisor_multi_turn_llm_merges_follow_up_fields() -> None:
    """多 Agent 多轮（LLM 路由恒 ticket_request）：第二轮补充字段必须与第一轮
    字段合并建单，previous_fields 不能因顶层缺持久化而丢失重新提取。"""
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-multi-turn-llm-001"}}

    first = graph.invoke(
        {"user_message": "我要申请退款，商品破损"},
        config=config,
    )
    assert first["needs_ticket"] is True
    assert first["missing_ticket_fields"] == ["order_id"]

    second = graph.invoke(
        {"user_message": "订单号是 A1001", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["intent"] == "ticket_request"
    assert second["missing_ticket_fields"] == []
    assert second["ticket_creation_status"] == "created"
    assert second["created_ticket"]["related_order_id"] == "A1001"
    assert "商品破损" in (second["created_ticket"]["description"] or "")


def test_supervisor_after_refund_execution_next_policy_turn_not_leaked() -> None:
    """退款执行成功后顶层 needs_ticket 残留 True 不应让下一轮知识问题被
    after_knowledge_agent 误转工单（knowledge 子图会重新计算 needs_ticket）。"""
    graph = build_supervisor_graph(
        router=RuleSupervisorRouter(),
        knowledge_service=FakePolicyRagService(
            make_policy_rag_answer(answer="退货政策是 30 天无理由。")
        ),
        ticket_creator=FakeTicketCreator(),
        refund_executor=FakeRefundExecutor(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=False,
    )
    config = {"configurable": {"thread_id": "supervisor-post-ticket-policy-001"}}

    first = graph.invoke(
        {"user_message": "我要申请退款，商品破损"},
        config=config,
    )
    assert first["missing_ticket_fields"] == ["order_id"]

    second = graph.invoke(
        {"user_message": "订单号是 A1001", "ticket_confirmation_approved": True},
        config=config,
    )
    assert second["refund_status"] == "succeeded"
    # 退款执行完成即流程终止：标志清除，防止同线程后续确认路由误执行退款
    assert second["refund_request_active"] is False

    third = graph.invoke(
        {"user_message": "退货政策是什么"},
        config=config,
    )
    assert third["rag_answer_status"] == "answered"
    assert "退货政策" in (third.get("final_answer") or "")
    assert third["needs_ticket"] is False
    # 未被误转工单：本轮没有再触发 create_ticket（created_ticket 保持上一轮值）
    assert third.get("ticket_creation_status") in (None, "created")


def test_supervisor_worker_short_circuits_on_angry() -> None:
    """愤怒消息在多 agent（supervisor）模式下经 worker 入口短路转人工。"""
    from app.agents.emotion_classifier import RuleBasedEmotionClassifier

    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.ORDER_QUERY),
        emotion_classifier=RuleBasedEmotionClassifier(),
    )
    result = graph.invoke(
        {
            "user_message": "气死我了！我的订单 202501010001 还不发货！",
            "normalized_message": "气死我了！我的订单 202501010001 还不发货！",
            "agent_trace_id": "t-1",
        }
    )
    assert result.get("emotion_handoff_requested") is True
    assert "人工客服" in result.get("final_answer", "")
