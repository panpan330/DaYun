import pytest

from app.agents.supervisor.supervisor_graph import build_supervisor_graph
from app.core.config import Settings
from app.core.exceptions import AppException
from app.services.console_agent_service import (
    ConsoleAgentActor,
    ConsoleAgentService,
)
from tests.tool_fakes import (
    FakePolicyRagService,
    FakeTicketCreator,
    make_created_ticket,
    make_policy_rag_answer,
)
from app.schemas.tool import QueryOrderArgs, QueryOrderResult
from app.schemas.console_agent import ConsoleAgentTicketFields


class _FakeConversationStore:
    """In-memory conversation store so console flows never touch real Redis."""

    def __init__(self) -> None:
        self.exchanges: list[dict[str, object]] = []

    def append_exchange(self, **kwargs: object) -> None:
        self.exchanges.append(kwargs)

    def get(self, *, actor, conversation_id: str):
        return None

    def close(self) -> None:
        pass


def _failing_order_executor(arguments: QueryOrderArgs) -> QueryOrderResult:
    raise AppException(
        code="ORDER_SERVICE_UNKNOWN",
        message="order service temporarily unavailable",
        status_code=500,
    )


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


def test_build_graph_returns_supervisor_graph_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(settings, graph=object())
    monkeypatch.setattr(
        service,
        "_create_redis_checkpointer",
        lambda: None,
    )
    graph = service._build_graph()
    assert graph is not None


def test_build_graph_returns_single_agent_graph_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, agent_multi_agent_enabled=False)
    service = ConsoleAgentService(settings, graph=object())
    monkeypatch.setattr(
        service,
        "_create_redis_checkpointer",
        lambda: None,
    )
    graph = service._build_graph()
    assert graph is not None


def test_multi_agent_console_end_to_end_rule_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: conversation -> supervisor route -> order agent -> answer."""
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )
    from app.agents.supervisor.supervisor_graph import build_supervisor_graph

    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.ORDER_QUERY),
        order_query_executor=_order_executor,
    )
    result = graph.invoke({"user_message": "查订单 A1001"})
    assert result["intent"] == "order_query"
    assert "已发货" in (result.get("final_answer") or "")


def test_multi_agent_console_decide_confirmation_resumes_and_creates_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """多 Agent 图确认中断后，console 确认流程不能因顶层 next 值误报 409。

    嵌套子图中断时顶层 next 为 ("ticket_agent",)，且顶层 state 不含
    ticket_fields（草稿位于顶层 interrupt payload）。修复前
    decide_ticket_confirmation 因 next 检查直接抛 409。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )

    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,
        conversation_store=_FakeConversationStore(),
    )
    actor = ConsoleAgentActor(
        user_id="U1001",
        tenant_id="default",
        roles=("customer",),
    )
    conversation_id = "conversation-ma-001"

    reply = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="我的订单 A1001 商品破损了，申请退款",
    )
    assert reply.pending_ticket_confirmation is not None
    confirmation_id = reply.pending_ticket_confirmation.confirmation_id

    response = service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=confirmation_id,
        approved=True,
    )
    assert response.created_ticket is not None
    assert response.created_ticket.requester_id == "U1001"


def test_multi_agent_console_order_failure_offers_human_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """多 Agent 模式订单查询失败必须保留"转人工"行为（human_handoff 非空）。

    修复前 SupervisorState 缺 order_query_* 字段，order 子图失败状态无法
    写回顶层，_human_handoff_from_state 读不到状态导致转人工恒为 None。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )

    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.ORDER_QUERY),
        order_query_executor=_failing_order_executor,
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,
        conversation_store=_FakeConversationStore(),
    )
    actor = ConsoleAgentActor(
        user_id="U1001",
        tenant_id="default",
        roles=("customer",),
    )
    conversation_id = "conversation-ma-002"

    response = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="查订单 A1001",
    )
    assert response.human_handoff is not None
    assert response.human_handoff.related_order_id == "A1001"


def test_multi_agent_console_second_confirmation_uses_fresh_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一会话第二张待确认工单不能复用上一轮顶层旧 ticket_fields。

    修复前 _pending_confirmation_fields 优先读顶层 values.ticket_fields（第一张
    工单完成后写回顶层），第二张工单中断时算出旧 confirmation_id，与活动中断
    不符，decide 误抛 TICKET_CONFIRMATION_MISMATCH 409。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )

    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,
        conversation_store=_FakeConversationStore(),
    )
    actor = ConsoleAgentActor(
        user_id="U1001",
        tenant_id="default",
        roles=("customer",),
    )
    conversation_id = "conversation-ma-003"

    first = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="我的订单 A1001 商品破损了，申请退款",
    )
    assert first.pending_ticket_confirmation is not None
    first_response = service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=first.pending_ticket_confirmation.confirmation_id,
        approved=True,
    )
    assert first_response.created_ticket is not None

    second = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="另一笔订单 B2002 也要申请售后",
    )
    assert second.pending_ticket_confirmation is not None
    # 两轮草稿必须不同（不同订单号），否则测试无法区分新旧草稿
    assert (
        second.pending_ticket_confirmation.confirmation_id
        != first.pending_ticket_confirmation.confirmation_id
    )
    second_response = service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=second.pending_ticket_confirmation.confirmation_id,
        approved=True,
    )
    assert second_response.created_ticket is not None
    assert second_response.created_ticket.related_order_id == "B2002"


def test_multi_agent_console_refund_confirmation_flag_and_transcript_copy() -> None:
    """退款执行确认必须暴露 is_refund_execution=True，且 transcript 文案按退款场景。

    后端 _record_exchange 硬编码"确认创建工单"会与前端"确认退款"弹窗文案不一致，
    刷新会话历史时文案打架；refund_request_active 是判别退款执行 vs 退款类型工单
    的唯一可靠标志（draft 字段两者完全相同）。
    """
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )
    from tests.tool_fakes import FakeRefundExecutor

    store = _FakeConversationStore()
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.REFUND_REQUEST),
        ticket_creator=FakeTicketCreator(),
        refund_executor=FakeRefundExecutor(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,
        conversation_store=store,
    )
    actor = ConsoleAgentActor(
        user_id="U1001",
        tenant_id="default",
        roles=("customer",),
    )
    conversation_id = "conversation-refund-flag-001"

    reply = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="我的订单 A1001 商品破损了，申请退款",
    )
    assert reply.pending_ticket_confirmation is not None
    assert reply.pending_ticket_confirmation.is_refund_execution is True
    confirmation_id = reply.pending_ticket_confirmation.confirmation_id

    # 修改信息：transcript 文案为退款版，且重新确认仍是退款确认
    corrected_response = service.correct_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=confirmation_id,
        ticket_fields=ConsoleAgentTicketFields(
            issue_type="refund",
            order_id="A1001",
            description="商品破损严重，申请全额退款",
            user_request="售后退款处理",
            urgency="normal",
            need_human_review=True,
        ),
    )
    assert store.exchanges[-1]["user_message"] == "修改退款信息并重新确认"
    assert corrected_response.pending_ticket_confirmation is not None
    assert corrected_response.pending_ticket_confirmation.is_refund_execution is True

    response = service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=corrected_response.pending_ticket_confirmation.confirmation_id,
        approved=True,
    )
    assert store.exchanges[-1]["user_message"] == "确认退款"
    assert response.pending_ticket_confirmation is None
    assert "退款" in response.reply


def test_multi_agent_console_ordinary_ticket_confirmation_not_refund() -> None:
    """普通工单流程（LLM 填 refund 类型）不得暴露退款执行标志或退款文案。"""
    from langgraph.checkpoint.memory import MemorySaver
    from app.agents.supervisor.supervisor_router import (
        FakeLLMSupervisorRouter,
        SupervisorRoute,
    )

    store = _FakeConversationStore()
    graph = build_supervisor_graph(
        router=FakeLLMSupervisorRouter(SupervisorRoute.TICKET_REQUEST),
        ticket_creator=FakeTicketCreator(),
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )
    settings = Settings(
        _env_file=None,
        agent_multi_agent_enabled=True,
        agent_mcp_tools_enabled=False,
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,
        conversation_store=store,
    )
    actor = ConsoleAgentActor(
        user_id="U1001",
        tenant_id="default",
        roles=("customer",),
    )
    conversation_id = "conversation-refund-flag-002"

    reply = service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message="订单 A1001 要退款，帮我建个工单投诉",
    )
    assert reply.pending_ticket_confirmation is not None
    assert reply.pending_ticket_confirmation.is_refund_execution is False

    response = service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=reply.pending_ticket_confirmation.confirmation_id,
        approved=True,
    )
    assert store.exchanges[-1]["user_message"] == "确认创建工单"
    assert response.created_ticket is not None
