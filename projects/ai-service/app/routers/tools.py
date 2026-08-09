import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header
from langchain_core.tools import StructuredTool

from app.core.config import Settings, get_settings
from app.schemas.tool import (
    LangChainToolInfo,
    LangChainToolListResponse,
    QueryOrderArgs,
    QueryOrderResponse,
    QueryOrderResult,
)
from app.schemas.tool_confirmation import (
    ConfirmToolConfirmationRequest,
    ToolConfirmationRequest,
    ToolConfirmationResponse,
)
from app.services.tool_confirmation_service import ToolConfirmationService
from app.tools.fake_order_tool import query_order as run_query_order_tool
from app.tools.langchain_tools import (
    create_query_order_langchain_tool,
    get_langchain_tool_metadata,
    list_model_callable_langchain_tools,
)
from app.tools.tool_confirmation import get_tool_confirmation_store
from app.tools.idempotency import IDEMPOTENCY_KEY_HEADER, run_idempotent_tool
from app.tools.tool_registry import authorize_tool_call


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


def get_tool_confirmation_service(
    settings: Settings = Depends(get_settings),
) -> ToolConfirmationService:
    return ToolConfirmationService(settings, get_tool_confirmation_store())


def get_query_order_langchain_tool(
    settings: Settings = Depends(get_settings),
) -> StructuredTool:
    return create_query_order_langchain_tool(settings=settings)


@router.post("/query-order", response_model=QueryOrderResponse)
def query_order(
    request: QueryOrderArgs,
    settings: Annotated[Settings, Depends(get_settings)],
    idempotency_key: Annotated[
        str | None,
        Header(alias=IDEMPOTENCY_KEY_HEADER),
    ] = None,
) -> QueryOrderResponse:
    authorize_tool_call("query_order")
    logger.info("fake_query_order_requested order_id=%s", request.order_id)
    result = run_idempotent_tool(
        "query_order",
        request,
        idempotency_key,
        lambda: run_query_order_tool(request, settings=settings),
    )
    return QueryOrderResponse(result=result)


@router.get("/langchain", response_model=LangChainToolListResponse)
def list_langchain_tools(
    settings: Settings = Depends(get_settings),
) -> LangChainToolListResponse:
    tools = list_model_callable_langchain_tools(settings=settings)
    return LangChainToolListResponse(
        tools=[
            LangChainToolInfo.model_validate(get_langchain_tool_metadata(tool))
            for tool in tools
        ]
    )


@router.post("/langchain/query-order", response_model=QueryOrderResponse)
def langchain_query_order(
    request: QueryOrderArgs,
    tool: StructuredTool = Depends(get_query_order_langchain_tool),
) -> QueryOrderResponse:
    logger.info(
        "langchain_query_order_requested order_id=%s",
        request.order_id,
    )
    raw_result = tool.invoke(request.model_dump(mode="json"))
    result = QueryOrderResult.model_validate(raw_result)
    return QueryOrderResponse(result=result)


@router.post("/confirmations", response_model=ToolConfirmationResponse)
def request_tool_confirmation(
    request: ToolConfirmationRequest,
    confirmation_service: ToolConfirmationService = Depends(
        get_tool_confirmation_service
    ),
) -> ToolConfirmationResponse:
    logger.info(
        "tool_confirmation_requested tool_name=%s actor_id=%s argument_count=%s",
        request.tool_name,
        request.actor_id,
        len(request.arguments),
    )
    return confirmation_service.request_confirmation(request)


@router.post(
    "/confirmations/{confirmation_id}/confirm",
    response_model=ToolConfirmationResponse,
)
def confirm_tool_confirmation(
    confirmation_id: str,
    request: ConfirmToolConfirmationRequest,
    confirmation_service: ToolConfirmationService = Depends(
        get_tool_confirmation_service
    ),
) -> ToolConfirmationResponse:
    logger.info(
        "tool_confirmation_confirmed confirmation_id=%s actor_id=%s",
        confirmation_id,
        request.actor_id,
    )
    return confirmation_service.confirm(confirmation_id, actor_id=request.actor_id)
