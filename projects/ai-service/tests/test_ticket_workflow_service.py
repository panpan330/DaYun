import logging

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketExtraction, TicketIntent
from app.schemas.ticket import (
    ExecuteTicketConfirmationRequest,
    TicketPlanRequest,
)
from app.schemas.tool_confirmation import ToolConfirmationRequest
from app.services.ticket_workflow_service import TicketWorkflowService
from app.services.tool_confirmation_service import ToolConfirmationService
from app.tools.tool_confirmation import ToolConfirmationStore
from tests.tool_fakes import (
    FakeTicketCreator,
    FakeTicketExtractor,
    make_ticket_extraction,
)


def make_extraction(
    *,
    intent: TicketIntent = TicketIntent.COMPLAINT,
) -> TicketExtraction:
    return make_ticket_extraction(intent=intent)


def make_workflow(
    extraction: TicketExtraction,
) -> tuple[TicketWorkflowService, ToolConfirmationService, FakeTicketCreator]:
    settings = Settings(_env_file=None)
    confirmation_service = ToolConfirmationService(
        settings,
        ToolConfirmationStore(),
    )
    ticket_creator = FakeTicketCreator()
    return (
        TicketWorkflowService(
            settings=settings,
            extractor=FakeTicketExtractor(extraction),
            confirmation_service=confirmation_service,
            ticket_creator=ticket_creator,
        ),
        confirmation_service,
        ticket_creator,
    )


def make_plan_request() -> TicketPlanRequest:
    return TicketPlanRequest(
        actor_id="demo_user_001",
        message="订单 A1001 已付款一周仍未发货，请帮我处理。",
    )


def test_plan_ticket_converts_validated_extraction_into_fixed_confirmation_plan() -> None:
    workflow, _, _ = make_workflow(make_extraction())

    plan = workflow.plan_ticket(make_plan_request())

    assert plan.extraction.intent == TicketIntent.COMPLAINT
    assert plan.confirmation.tool_name == "create_ticket"
    assert plan.confirmation.arguments == {
        "requester_id": "demo_user_001",
        "title": "订单 A1001 一直未发货",
        "description": "订单 A1001 已付款一周仍未发货，请帮我处理。",
        "category": "complaint",
        "priority": "high",
        "related_order_id": "A1001",
    }
    assert plan.confirmation.status == "pending"


def test_plan_ticket_rejects_unknown_intent_instead_of_guessing_business_category() -> None:
    workflow, _, _ = make_workflow(make_extraction(intent=TicketIntent.UNKNOWN))

    with pytest.raises(AppException) as exc_info:
        workflow.plan_ticket(make_plan_request())

    assert exc_info.value.code == "TICKET_INTENT_UNSUPPORTED"
    assert exc_info.value.status_code == 422


def test_execute_ticket_requires_confirmed_plan_and_reuses_result_on_retry() -> None:
    workflow, confirmation_service, ticket_creator = make_workflow(make_extraction())
    plan = workflow.plan_ticket(make_plan_request())
    execute_request = ExecuteTicketConfirmationRequest(actor_id="demo_user_001")

    with pytest.raises(AppException) as pending_error:
        workflow.execute_confirmed_ticket(
            plan.confirmation.confirmation_id,
            execute_request,
        )
    assert pending_error.value.code == "TOOL_CONFIRMATION_REQUIRED"

    confirmation_service.confirm(
        plan.confirmation.confirmation_id,
        actor_id="demo_user_001",
    )
    first_result = workflow.execute_confirmed_ticket(
        plan.confirmation.confirmation_id,
        execute_request,
    )
    second_result = workflow.execute_confirmed_ticket(
        plan.confirmation.confirmation_id,
        execute_request,
    )

    assert first_result.ticket.ticket_id == "T1001"
    assert second_result == first_result
    assert len(ticket_creator.calls) == 1


def test_execute_ticket_logs_key_stages_without_full_description(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workflow, confirmation_service, _ = make_workflow(make_extraction())
    plan = workflow.plan_ticket(make_plan_request())
    confirmation_service.confirm(
        plan.confirmation.confirmation_id,
        actor_id="demo_user_001",
    )
    caplog.set_level(logging.INFO, logger="app.services.ticket_workflow_service")

    workflow.execute_confirmed_ticket(
        plan.confirmation.confirmation_id,
        ExecuteTicketConfirmationRequest(actor_id="demo_user_001"),
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.services.ticket_workflow_service"
    ]

    assert any(
        message.startswith("ticket_confirmation_loaded confirmation_id=")
        for message in messages
    )
    assert any(
        message.startswith("ticket_tool_execution_started confirmation_id=")
        for message in messages
    )
    assert any("ticket_id=T1001" in message for message in messages)
    assert all("已付款一周仍未发货" not in message for message in messages)


def test_execute_ticket_rechecks_actor_and_stored_argument_contract() -> None:
    workflow, confirmation_service, ticket_creator = make_workflow(make_extraction())
    pending = confirmation_service.request_confirmation(
        ToolConfirmationRequest(
            actor_id="demo_user_001",
            tool_name="create_ticket",
            arguments={"title": "缺少创建工单所需字段"},
        )
    )
    confirmation_service.confirm(pending.confirmation_id, actor_id="demo_user_001")

    with pytest.raises(AppException) as actor_error:
        workflow.execute_confirmed_ticket(
            pending.confirmation_id,
            ExecuteTicketConfirmationRequest(actor_id="other_user_002"),
        )
    assert actor_error.value.code == "TOOL_CONFIRMATION_FORBIDDEN"

    with pytest.raises(AppException) as argument_error:
        workflow.execute_confirmed_ticket(
            pending.confirmation_id,
            ExecuteTicketConfirmationRequest(actor_id="demo_user_001"),
        )
    assert argument_error.value.code == "TICKET_ARGUMENTS_VALIDATION_FAILED"
    assert ticket_creator.calls == []
