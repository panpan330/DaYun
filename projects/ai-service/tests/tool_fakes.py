from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.core.exceptions import AppException
from app.rag.generator import (
    RagAnswer,
    build_grounded_rag_answer,
    build_no_context_rag_answer,
)
from app.schemas.cancel import CancelOrderArgs
from app.schemas.refund import RefundOrderArgs
from app.schemas.structured import TicketExtraction, TicketIntent, TicketUrgency
from app.schemas.ticket import CreateTicketArgs, CreatedTicket
from tests.rag_fakes import make_retrieved_chunk


def make_java_order_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "order_id": "A1001",
        "customer_id": "C9001",
        "order_status": "waiting_shipment",
        "payment_status": "paid",
        "logistics_message": "商家已接单，等待仓库发货。",
        "latest_event": "仓库正在准备出库。",
        "can_create_ticket": True,
    }
    payload.update(overrides)
    return payload


class FakeOrderLookupClient:
    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload or make_java_order_payload()
        self.error = error
        self.calls: list[str] = []

    def get_order(self, order_id: str) -> Mapping[str, Any]:
        self.calls.append(order_id)
        if self.error is not None:
            raise self.error
        return self.payload


def make_ticket_extraction(
    *,
    intent: TicketIntent = TicketIntent.COMPLAINT,
    order_id: str | None = "A1001",
    summary: str = "订单 A1001 一直未发货",
    urgency: TicketUrgency = TicketUrgency.HIGH,
    need_human_review: bool = True,
) -> TicketExtraction:
    return TicketExtraction(
        intent=intent,
        order_id=order_id,
        summary=summary,
        urgency=urgency,
        need_human_review=need_human_review,
    )


class FakeTicketExtractor:
    def __init__(self, extraction: TicketExtraction | None = None) -> None:
        self.extraction = extraction or make_ticket_extraction()
        self.messages: list[str] = []

    def extract_ticket(self, user_message: str) -> TicketExtraction:
        self.messages.append(user_message)
        return self.extraction


def make_policy_rag_answer(
    *,
    answer: str = "根据测试知识库，退款通常需要核对订单状态和售后条件。",
) -> RagAnswer:
    return build_grounded_rag_answer(
        answer,
        [
            make_retrieved_chunk(
                chunk_id="fake_policy_chunk_0001",
                content="退款通常需要核对订单状态和售后条件。",
                metadata={
                    "source": "fake-policy.md",
                    "title": "测试政策",
                    "section": "退款",
                    "chunk_id": "fake_policy_chunk_0001",
                },
            )
        ],
    )


class FakePolicyRagService:
    def __init__(
        self,
        answer: RagAnswer | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.answer = answer or make_policy_rag_answer()
        self.error = error
        self.queries: list[str] = []

    def answer_policy_question(self, query: str, *, access_scope=None) -> RagAnswer:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.answer


class FakeNoContextPolicyRagService(FakePolicyRagService):
    def __init__(self) -> None:
        super().__init__(build_no_context_rag_answer())


def make_created_ticket(
    arguments: CreateTicketArgs | None = None,
    *,
    ticket_id: str = "T1001",
    created_at: datetime | None = None,
) -> CreatedTicket:
    if arguments is None:
        arguments = CreateTicketArgs(
            requester_id="demo_user_001",
            title="订单 A1001 一直未发货",
            description="订单 A1001 已付款一周仍未发货，请帮我处理。",
            category="complaint",
            priority="high",
            related_order_id="A1001",
        )

    return CreatedTicket(
        ticket_id=ticket_id,
        requester_id=arguments.requester_id,
        title=arguments.title,
        description=arguments.description,
        category=arguments.category,
        priority=arguments.priority,
        related_order_id=arguments.related_order_id,
        created_at=created_at
        or datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc),
    )


class FakeTicketCreator:
    def __init__(
        self,
        *,
        ticket: CreatedTicket | None = None,
        error: Exception | None = None,
    ) -> None:
        self.ticket = ticket
        self.error = error
        self.calls: list[CreateTicketArgs] = []
        self.idempotency_keys: list[str] = []

    def create_ticket(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket:
        self.calls.append(arguments)
        self.idempotency_keys.append(idempotency_key)
        if self.error is not None:
            raise self.error
        return self.ticket or make_created_ticket(arguments)


class FakeRefundExecutor:
    """In-memory RefundExecutor used to test the refund flow without Java/MCP."""

    def __init__(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[RefundOrderArgs] = []
        self.idempotency_keys: list[str] = []

    def refund_order(
        self,
        arguments: RefundOrderArgs,
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        self.calls.append(arguments)
        self.idempotency_keys.append(idempotency_key)
        if self.error is not None:
            raise self.error
        return self.result or {
            "order_id": arguments.order_id,
            "refund_status": "succeeded",
        }


class FakeCancelExecutor:
    """In-memory CancelExecutor used to test the cancel flow without Java/MCP."""

    def __init__(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[CancelOrderArgs] = []
        self.idempotency_keys: list[str] = []

    def cancel_order(
        self,
        arguments: CancelOrderArgs,
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        self.calls.append(arguments)
        self.idempotency_keys.append(idempotency_key)
        if self.error is not None:
            raise self.error
        return self.result or {
            "order_id": arguments.order_id,
            "order_status": "canceled",
        }


class FakeMcpToolCaller:
    """In-memory McpToolCaller used to test agent adapters without HTTP."""

    def __init__(self, responses: dict[str, dict] | None = None) -> None:
        self.responses: dict[str, dict] = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        self.calls.append((tool_name, arguments))
        if tool_name in self.responses:
            return self.responses[tool_name]
        raise AppException(
            code="MCP_TOOL_NOT_FOUND",
            message=f"tool {tool_name} not available",
            status_code=404,
        )
