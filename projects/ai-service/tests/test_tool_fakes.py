import pytest

from app.core.exceptions import AppException
from app.schemas.structured import TicketIntent
from app.schemas.ticket import CreateTicketArgs
from tests.tool_fakes import (
    FakeNoContextPolicyRagService,
    FakeOrderLookupClient,
    FakePolicyRagService,
    FakeTicketCreator,
    FakeTicketExtractor,
    make_created_ticket,
    make_java_order_payload,
    make_policy_rag_answer,
    make_ticket_extraction,
)


def test_fake_order_lookup_client_returns_payload_and_records_calls() -> None:
    client = FakeOrderLookupClient(make_java_order_payload(order_id="A1002"))

    payload = client.get_order("A1002")

    assert payload["order_id"] == "A1002"
    assert client.calls == ["A1002"]


def test_fake_order_lookup_client_can_raise_configured_error() -> None:
    client = FakeOrderLookupClient(error=RuntimeError("java unavailable"))

    with pytest.raises(RuntimeError, match="java unavailable"):
        client.get_order("A1001")

    assert client.calls == ["A1001"]


def test_fake_ticket_extractor_returns_configured_extraction() -> None:
    extraction = make_ticket_extraction(intent=TicketIntent.LOGISTICS)
    extractor = FakeTicketExtractor(extraction)

    result = extractor.extract_ticket("订单 A1001 物流卡住了")

    assert result.intent == TicketIntent.LOGISTICS
    assert extractor.messages == ["订单 A1001 物流卡住了"]


def test_make_policy_rag_answer_returns_grounded_answer_with_citation() -> None:
    answer = make_policy_rag_answer(answer="测试回答")

    assert answer.status.value == "answered"
    assert answer.answer == "测试回答"
    assert answer.citations[0].source == "fake-policy.md"
    assert answer.citations[0].chunk_id == "fake_policy_chunk_0001"


def test_fake_policy_rag_service_returns_answer_and_records_queries() -> None:
    service = FakePolicyRagService(make_policy_rag_answer(answer="固定政策回答"))

    answer = service.answer_policy_question("退款规则是什么？")

    assert answer.answer == "固定政策回答"
    assert answer.status.value == "answered"
    assert service.queries == ["退款规则是什么？"]


def test_fake_policy_rag_service_can_raise_configured_error() -> None:
    service = FakePolicyRagService(error=RuntimeError("rag unavailable"))

    with pytest.raises(RuntimeError, match="rag unavailable"):
        service.answer_policy_question("退款规则是什么？")

    assert service.queries == ["退款规则是什么？"]


def test_fake_no_context_policy_rag_service_returns_no_context() -> None:
    service = FakeNoContextPolicyRagService()

    answer = service.answer_policy_question("会员等级政策是什么？")

    assert answer.status.value == "no_context"
    assert answer.citations == []
    assert service.queries == ["会员等级政策是什么？"]


def test_fake_ticket_creator_returns_ticket_and_records_idempotency_key() -> None:
    creator = FakeTicketCreator()
    arguments = CreateTicketArgs(
        requester_id="demo_user_001",
        title="订单 A1001 一直未发货",
        description="订单 A1001 已付款一周仍未发货，请帮我处理。",
        category="complaint",
        priority="high",
        related_order_id="A1001",
    )

    ticket = creator.create_ticket(
        arguments,
        idempotency_key="confirmation-idempotency-001",
    )

    assert ticket == make_created_ticket(arguments)
    assert creator.calls == [arguments]
    assert creator.idempotency_keys == ["confirmation-idempotency-001"]


def test_fake_ticket_creator_can_raise_configured_app_exception() -> None:
    error = AppException(
        code="TOOL_UPSTREAM_ERROR",
        message="工单业务服务暂时不可用，请稍后重试。",
        status_code=502,
    )
    creator = FakeTicketCreator(error=error)

    with pytest.raises(AppException) as exc_info:
        creator.create_ticket(
            CreateTicketArgs(
                requester_id="demo_user_001",
                title="订单 A1001 一直未发货",
                description="订单 A1001 已付款一周仍未发货，请帮我处理。",
                category="complaint",
                priority="high",
                related_order_id="A1001",
            ),
            idempotency_key="confirmation-idempotency-002",
        )

    assert exc_info.value is error
    assert creator.idempotency_keys == ["confirmation-idempotency-002"]
