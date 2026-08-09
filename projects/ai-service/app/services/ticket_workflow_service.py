import logging
from collections.abc import Mapping
from time import perf_counter
from typing import Protocol

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketExtraction, TicketIntent, TicketUrgency
from app.schemas.ticket import (
    CreateTicketArgs,
    CreatedTicket,
    ExecuteTicketConfirmationRequest,
    TicketCategory,
    TicketExecutionResponse,
    TicketPlanRequest,
    TicketPlanResponse,
    TicketPriority,
)
from app.schemas.tool_confirmation import ToolConfirmationRequest
from app.services.java_ticket_client import JavaTicketClient
from app.services.structured_output_service import (
    create_structured_output_service,
)
from app.services.tool_confirmation_service import ToolConfirmationService
from app.tools.idempotency import run_idempotent_tool
from app.tools.tool_confirmation import get_tool_confirmation_store
from app.tools.tool_registry import authorize_tool_call


logger = logging.getLogger(__name__)


class TicketExtractor(Protocol):
    def extract_ticket(self, user_message: str) -> TicketExtraction: ...


class TicketCreator(Protocol):
    def create_ticket(
        self,
        arguments: CreateTicketArgs,
        *,
        idempotency_key: str,
    ) -> CreatedTicket: ...


_INTENT_TO_CATEGORY: Mapping[TicketIntent, TicketCategory] = {
    TicketIntent.REFUND: TicketCategory.REFUND,
    TicketIntent.ORDER_QUERY: TicketCategory.ORDER_QUERY,
    TicketIntent.LOGISTICS: TicketCategory.LOGISTICS,
    TicketIntent.COMPLAINT: TicketCategory.COMPLAINT,
}

_URGENCY_TO_PRIORITY: Mapping[TicketUrgency, TicketPriority] = {
    TicketUrgency.LOW: TicketPriority.LOW,
    TicketUrgency.NORMAL: TicketPriority.NORMAL,
    TicketUrgency.HIGH: TicketPriority.HIGH,
}


def build_create_ticket_args(
    extraction: TicketExtraction,
    *,
    actor_id: str,
    original_message: str,
) -> CreateTicketArgs:
    """Convert model output into a backend-owned, validated business command."""

    category = _INTENT_TO_CATEGORY.get(extraction.intent)
    if category is None:
        raise AppException(
            code="TICKET_INTENT_UNSUPPORTED",
            message="暂时无法确定应创建哪类工单，请补充具体问题后再试。",
            status_code=422,
        )

    return CreateTicketArgs(
        requester_id=actor_id,
        title=extraction.summary,
        description=original_message,
        category=category,
        priority=_URGENCY_TO_PRIORITY[extraction.urgency],
        related_order_id=extraction.order_id,
    )


class TicketWorkflowService:
    def __init__(
        self,
        *,
        settings: Settings,
        extractor: TicketExtractor,
        confirmation_service: ToolConfirmationService,
        ticket_creator: TicketCreator,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.confirmation_service = confirmation_service
        self.ticket_creator = ticket_creator

    def plan_ticket(self, request: TicketPlanRequest) -> TicketPlanResponse:
        extraction = self.extractor.extract_ticket(request.message)
        arguments = build_create_ticket_args(
            extraction,
            actor_id=request.actor_id,
            original_message=request.message,
        )
        logger.info(
            (
                "ticket_plan_arguments_built actor_id=%s intent=%s category=%s "
                "priority=%s has_related_order_id=%s need_human_review=%s"
            ),
            request.actor_id,
            extraction.intent.value,
            arguments.category.value,
            arguments.priority.value,
            arguments.related_order_id is not None,
            extraction.need_human_review,
        )
        confirmation = self.confirmation_service.request_confirmation(
            ToolConfirmationRequest(
                actor_id=request.actor_id,
                tool_name="create_ticket",
                arguments=arguments.model_dump(mode="json"),
            )
        )
        logger.info(
            "ticket_plan_created confirmation_id=%s actor_id=%s category=%s priority=%s",
            confirmation.confirmation_id,
            request.actor_id,
            arguments.category,
            arguments.priority,
        )
        return TicketPlanResponse(
            extraction=extraction,
            confirmation=confirmation,
        )

    def execute_confirmed_ticket(
        self,
        confirmation_id: str,
        request: ExecuteTicketConfirmationRequest,
    ) -> TicketExecutionResponse:
        start_time = perf_counter()
        try:
            record = self.confirmation_service.require_confirmed(
                confirmation_id,
                actor_id=request.actor_id,
            )
            logger.info(
                (
                    "ticket_confirmation_loaded confirmation_id=%s actor_id=%s "
                    "tool_name=%s"
                ),
                confirmation_id,
                request.actor_id,
                record.tool_name,
            )
            if record.tool_name != "create_ticket":
                raise AppException(
                    code="TOOL_CONFIRMATION_TOOL_MISMATCH",
                    message="该确认单不属于创建工单操作。",
                    status_code=409,
                )

            try:
                arguments = CreateTicketArgs.model_validate(record.arguments)
            except ValidationError as exc:
                raise AppException(
                    code="TICKET_ARGUMENTS_VALIDATION_FAILED",
                    message="确认单中的工单参数不符合创建工单契约。",
                    status_code=422,
                    details=exc.errors(include_url=False),
                ) from exc

            authorize_tool_call("create_ticket", user_confirmed=True)
            logger.info(
                (
                    "ticket_tool_execution_started confirmation_id=%s actor_id=%s "
                    "category=%s priority=%s related_order_id=%s"
                ),
                confirmation_id,
                request.actor_id,
                arguments.category.value,
                arguments.priority.value,
                arguments.related_order_id,
            )
            ticket = run_idempotent_tool(
                "create_ticket",
                arguments,
                confirmation_id,
                lambda: self.ticket_creator.create_ticket(
                    arguments,
                    idempotency_key=confirmation_id,
                ),
            )
        except AppException as exc:
            logger.warning(
                (
                    "ticket_execution_failed confirmation_id=%s actor_id=%s "
                    "code=%s status_code=%s elapsed_ms=%.2f"
                ),
                confirmation_id,
                request.actor_id,
                exc.code,
                exc.status_code,
                (perf_counter() - start_time) * 1000,
            )
            raise

        logger.info(
            (
                "ticket_created_from_confirmation confirmation_id=%s actor_id=%s "
                "ticket_id=%s elapsed_ms=%.2f"
            ),
            confirmation_id,
            request.actor_id,
            ticket.ticket_id,
            (perf_counter() - start_time) * 1000,
        )
        return TicketExecutionResponse(
            confirmation_id=confirmation_id,
            ticket=ticket,
        )


def create_ticket_workflow_service(settings: Settings) -> TicketWorkflowService:
    return TicketWorkflowService(
        settings=settings,
        extractor=create_structured_output_service(settings),
        confirmation_service=ToolConfirmationService(
            settings,
            get_tool_confirmation_store(),
        ),
        ticket_creator=JavaTicketClient.from_settings(settings),
    )
