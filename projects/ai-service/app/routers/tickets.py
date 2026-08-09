import logging

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.ticket import (
    ExecuteTicketConfirmationRequest,
    TicketExecutionResponse,
    TicketPlanRequest,
    TicketPlanResponse,
)
from app.services.ticket_workflow_service import (
    TicketWorkflowService,
    create_ticket_workflow_service,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])


def get_ticket_workflow_service(
    settings: Settings = Depends(get_settings),
) -> TicketWorkflowService:
    return create_ticket_workflow_service(settings)


@router.post("/plans", response_model=TicketPlanResponse)
def plan_ticket(
    request: TicketPlanRequest,
    workflow_service: TicketWorkflowService = Depends(get_ticket_workflow_service),
) -> TicketPlanResponse:
    logger.info(
        "ticket_plan_requested actor_id=%s message_length=%s",
        request.actor_id,
        len(request.message),
    )
    return workflow_service.plan_ticket(request)


@router.post(
    "/confirmations/{confirmation_id}/execute",
    response_model=TicketExecutionResponse,
    status_code=201,
)
def execute_ticket_confirmation(
    confirmation_id: str,
    request: ExecuteTicketConfirmationRequest,
    workflow_service: TicketWorkflowService = Depends(get_ticket_workflow_service),
) -> TicketExecutionResponse:
    logger.info(
        "ticket_execution_requested confirmation_id=%s actor_id=%s",
        confirmation_id,
        request.actor_id,
    )
    return workflow_service.execute_confirmed_ticket(confirmation_id, request)
