import json
import logging
from contextvars import Context, copy_context
from uuid import uuid4
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.core.ai_security_boundary import (
    redact_sensitive_text,
    require_prompt_injection_safe,
)
from app.core.business_context import reset_business_context, set_business_context
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.core.trace import get_trace_id
from app.schemas.chat import ChatRequest, ChatResponse, ConsoleChatRequest, ConsoleChatResponse
from app.schemas.console_agent import (
    ConsoleAgentConversation,
    ConsoleAgentConversationSummary,
    ConsoleAgentFeedbackRequest,
    ConsoleAgentFeedbackResponse,
    ConsoleAgentConfirmationRequest,
    ConsoleAgentMessageRequest,
    ConsoleAgentResponse,
    ConsoleAgentTicketCorrectionRequest,
    ConversationSummaryView,
    HumanReplyRequest,
)
from app.schemas.structured import StructuredOutputRequest, StructuredOutputResponse
from app.schemas.tool_decision import ToolDecisionResponse
from app.services.llm_service import LLMChatService, create_llm_chat_service
from app.services.langchain_chat_model_service import (
    LangChainChatModelService,
    create_langchain_chat_model_service,
)
from app.services.langchain_structured_output_service import (
    LangChainStructuredOutputService,
    create_langchain_structured_output_service,
)
from app.services.structured_output_service import (
    StructuredOutputService,
    create_structured_output_service,
)
from app.services.tool_decision_service import (
    ToolDecisionService,
    create_tool_decision_service,
)
from app.services.tool_calling_chat_service import (
    ToolCallingChatService,
    create_tool_calling_chat_service,
)
from app.services.console_agent_service import (
    ConsoleAgentActor,
    ConsoleAgentService,
    JavaConsoleAgentActorResolver,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
SSE_MEDIA_TYPE = "text/event-stream"
SSE_RECONNECT_DELAY_MS = 3000


class ContextBoundAgentEventIterator:
    """Keep one Agent generator in the ContextVar context that created its tokens."""

    def __init__(self, events: Iterator[dict[str, Any]]) -> None:
        self._events = events
        self._context: Context = copy_context()

    def __iter__(self) -> "ContextBoundAgentEventIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        return self._context.run(next, self._events)


def get_llm_chat_service(
    settings: Settings = Depends(get_settings),
) -> LLMChatService:
    return create_llm_chat_service(settings)


def get_langchain_chat_model_service(
    settings: Settings = Depends(get_settings),
) -> LangChainChatModelService:
    return create_langchain_chat_model_service(settings)


def get_langchain_structured_output_service(
    settings: Settings = Depends(get_settings),
) -> LangChainStructuredOutputService:
    return create_langchain_structured_output_service(settings)


def get_structured_output_service(
    settings: Settings = Depends(get_settings),
) -> StructuredOutputService:
    return create_structured_output_service(settings)


def get_tool_decision_service(
    settings: Settings = Depends(get_settings),
) -> ToolDecisionService:
    return create_tool_decision_service(settings)


def get_tool_calling_chat_service(
    settings: Settings = Depends(get_settings),
) -> ToolCallingChatService:
    return create_tool_calling_chat_service(settings)


def get_console_agent_service(http_request: Request) -> ConsoleAgentService:
    return http_request.app.state.console_agent_service


def get_console_agent_actor(
    http_request: Request,
    settings: Settings = Depends(get_settings),
) -> ConsoleAgentActor:
    return JavaConsoleAgentActorResolver(settings).resolve(
        http_request.headers.get("Authorization")
    )


def format_sse_event(
    event: str,
    data: dict[str, object],
    *,
    event_id: str | None = None,
    retry_ms: int | None = None,
) -> str:
    json_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json_data}")
    return "\n".join(lines) + "\n\n"


def format_sse_comment(comment: str) -> str:
    return f": {comment}\n\n"


def build_sse_headers(*, trace_id: str) -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Trace-Id": trace_id,
    }


def validate_chat_request_security(request: ChatRequest) -> None:
    require_prompt_injection_safe(request.message, source="user")
    for message in request.history:
        require_prompt_injection_safe(message.content, source="history")


async def build_stream_events(
    chunks: Iterator[str],
    *,
    trace_id: str,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    heartbeat_every_chunks: int = 0,
) -> AsyncIterator[str]:
    yield format_sse_event(
        "start",
        {"trace_id": trace_id},
        event_id=f"{trace_id}:0",
        retry_ms=SSE_RECONNECT_DELAY_MS,
    )
    chunk_count = 0
    try:
        async for chunk in iterate_in_threadpool(chunks):
            if is_disconnected is not None and await is_disconnected():
                logger.info(
                    "stream_chat_client_disconnected trace_id=%s chunks=%s",
                    trace_id,
                    chunk_count,
                )
                return

            chunk_count += 1
            yield format_sse_event(
                "message",
                {"content": redact_sensitive_text(chunk)},
                event_id=f"{trace_id}:{chunk_count}",
            )
            if heartbeat_every_chunks > 0 and chunk_count % heartbeat_every_chunks == 0:
                yield format_sse_comment("heartbeat")
    except AppException as exc:
        logger.warning(
            "stream_chat_app_exception code=%s",
            exc.code,
        )
        yield format_sse_event(
            "error",
            {
                "code": exc.code,
                "message": exc.message,
                "trace_id": trace_id,
            },
            event_id=f"{trace_id}:error",
        )
        return
    except Exception:
        logger.exception("stream_chat_unhandled_exception")
        yield format_sse_event(
            "error",
            {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "服务器内部错误",
                "trace_id": trace_id,
            },
            event_id=f"{trace_id}:error",
        )
        return

    yield format_sse_event(
        "done",
        {"trace_id": trace_id, "chunks": chunk_count},
        event_id=f"{trace_id}:done",
    )


async def build_console_agent_stream_events(
    events: Iterator[dict[str, Any]],
    *,
    trace_id: str,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    event_index = 0
    context_bound_events = ContextBoundAgentEventIterator(events)
    async for event in iterate_in_threadpool(context_bound_events):
        if await is_disconnected():
            logger.info("console_agent_stream_client_disconnected trace_id=%s", trace_id)
            return
        event_name = event.get("event")
        event_data = event.get("data")
        if not isinstance(event_name, str) or not isinstance(event_data, dict):
            continue
        event_index += 1
        yield format_sse_event(
            event_name,
            event_data,
            event_id=f"{trace_id}:{event_index}",
            retry_ms=SSE_RECONNECT_DELAY_MS if event_name == "start" else None,
        )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    llm_chat_service: LLMChatService = Depends(get_llm_chat_service),
) -> ChatResponse:
    validate_chat_request_security(request)
    logger.info(
        "chat_requested message_length=%s history_size=%s",
        len(request.message),
        len(request.history),
    )
    reply = llm_chat_service.generate_reply(
        request.message,
        history=request.history,
    )
    return ChatResponse(reply=redact_sensitive_text(reply))


@router.post("/langchain-chat", response_model=ChatResponse)
def langchain_chat(
    request: ChatRequest,
    langchain_chat_model_service: LangChainChatModelService = Depends(
        get_langchain_chat_model_service
    ),
) -> ChatResponse:
    validate_chat_request_security(request)
    logger.info(
        "langchain_chat_requested message_length=%s history_size=%s",
        len(request.message),
        len(request.history),
    )
    reply = langchain_chat_model_service.generate_reply(
        request.message,
        history=request.history,
    )
    return ChatResponse(reply=redact_sensitive_text(reply))


@router.post("/stream-chat")
def stream_chat(
    request: Request,
    chat_request: ChatRequest,
    settings: Settings = Depends(get_settings),
    llm_chat_service: LLMChatService = Depends(get_llm_chat_service),
) -> StreamingResponse:
    validate_chat_request_security(chat_request)
    logger.info(
        "stream_chat_requested message_length=%s history_size=%s",
        len(chat_request.message),
        len(chat_request.history),
    )
    chunks = llm_chat_service.stream_reply(
        chat_request.message,
        history=chat_request.history,
    )
    trace_id = get_trace_id()
    return StreamingResponse(
        build_stream_events(
            chunks,
            trace_id=trace_id,
            is_disconnected=request.is_disconnected,
            heartbeat_every_chunks=settings.sse_heartbeat_every_chunks,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers=build_sse_headers(trace_id=trace_id),
    )


@router.post("/extract-ticket", response_model=StructuredOutputResponse)
def extract_ticket(
    request: StructuredOutputRequest,
    structured_output_service: StructuredOutputService = Depends(
        get_structured_output_service
    ),
) -> StructuredOutputResponse:
    require_prompt_injection_safe(request.message, source="user")
    logger.info(
        "extract_ticket_requested message_length=%s",
        len(request.message),
    )
    extraction = structured_output_service.extract_ticket(request.message)
    return StructuredOutputResponse(extraction=extraction)


@router.post("/langchain-extract-ticket", response_model=StructuredOutputResponse)
def langchain_extract_ticket(
    request: StructuredOutputRequest,
    langchain_structured_output_service: LangChainStructuredOutputService = Depends(
        get_langchain_structured_output_service
    ),
) -> StructuredOutputResponse:
    require_prompt_injection_safe(request.message, source="user")
    logger.info(
        "langchain_extract_ticket_requested message_length=%s",
        len(request.message),
    )
    extraction = langchain_structured_output_service.extract_ticket(request.message)
    return StructuredOutputResponse(extraction=extraction)


@router.post("/tool-decision", response_model=ToolDecisionResponse)
def tool_decision(
    request: ChatRequest,
    tool_decision_service: ToolDecisionService = Depends(get_tool_decision_service),
) -> ToolDecisionResponse:
    validate_chat_request_security(request)
    logger.info(
        "tool_decision_requested message_length=%s history_size=%s",
        len(request.message),
        len(request.history),
    )
    return tool_decision_service.decide(
        request.message,
        history=request.history,
    )


@router.post("/tool-chat", response_model=ChatResponse)
def tool_chat(
    http_request: Request,
    request: ChatRequest,
    tool_calling_chat_service: ToolCallingChatService = Depends(
        get_tool_calling_chat_service
    ),
) -> ChatResponse:
    validate_chat_request_security(request)
    logger.info(
        "tool_chat_requested message_length=%s history_size=%s",
        len(request.message),
        len(request.history),
    )
    context_tokens = set_business_context(
        user_id=http_request.headers.get("X-User-Id"),
        tenant_id=http_request.headers.get("X-Tenant-Id"),
    )
    try:
        reply = tool_calling_chat_service.generate_reply(
            request.message,
            history=request.history,
        )
    finally:
        reset_business_context(context_tokens)
    return ChatResponse(reply=redact_sensitive_text(reply))


@router.post("/api/ai/chat", response_model=ConsoleChatResponse)
def console_ai_chat(
    http_request: Request,
    request: ConsoleChatRequest,
    tool_calling_chat_service: ToolCallingChatService = Depends(
        get_tool_calling_chat_service
    ),
) -> ConsoleChatResponse:
    validate_chat_request_security(request)
    conversation_id = request.conversation_id or f"local-{uuid4().hex}"
    trace_id = get_trace_id()
    logger.info(
        "console_ai_chat_requested conversation_id=%s message_length=%s history_size=%s",
        conversation_id,
        len(request.message),
        len(request.history),
    )
    context_tokens = set_business_context(
        user_id=http_request.headers.get("X-User-Id"),
        tenant_id=http_request.headers.get("X-Tenant-Id"),
    )
    try:
        reply = tool_calling_chat_service.generate_reply(
            request.message,
            history=request.history,
        )
    finally:
        reset_business_context(context_tokens)
    return ConsoleChatResponse(
        reply=redact_sensitive_text(reply),
        conversation_id=conversation_id,
        trace_id=trace_id,
        mode="tool_chat",
    )


@router.post("/api/ai/agent/conversations", response_model=ConsoleAgentResponse)
def console_agent_chat(
    request: ConsoleAgentMessageRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentResponse:
    validate_chat_request_security(request)
    conversation_id = request.conversation_id or f"agent-{uuid4().hex}"
    logger.info(
        "console_agent_requested conversation_id=%s actor_id=%s message_length=%s history_size=%s",
        conversation_id,
        actor.user_id,
        len(request.message),
        len(request.history),
    )
    return agent_service.reply(
        actor=actor,
        conversation_id=conversation_id,
        message=request.message,
    )


@router.post("/api/ai/agent/conversations/stream")
def stream_console_agent_chat(
    http_request: Request,
    request: ConsoleAgentMessageRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> StreamingResponse:
    validate_chat_request_security(request)
    conversation_id = request.conversation_id or f"agent-{uuid4().hex}"
    trace_id = get_trace_id()
    logger.info(
        "console_agent_stream_requested conversation_id=%s actor_id=%s message_length=%s",
        conversation_id,
        actor.user_id,
        len(request.message),
    )
    events = agent_service.stream_reply(
        actor=actor,
        conversation_id=conversation_id,
        message=request.message,
        trace_id=trace_id,
    )
    return StreamingResponse(
        build_console_agent_stream_events(
            events,
            trace_id=trace_id,
            is_disconnected=http_request.is_disconnected,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers=build_sse_headers(trace_id=trace_id),
    )


@router.get(
    "/api/ai/agent/conversations",
    response_model=list[ConsoleAgentConversationSummary],
)
def list_console_agent_conversations(
    limit: int = Query(default=20, ge=1, le=30),
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> list[ConsoleAgentConversationSummary]:
    return agent_service.list_conversations(actor=actor, limit=limit)


@router.get(
    "/api/ai/agent/conversations/{conversation_id}/history",
    response_model=ConsoleAgentConversation,
)
def get_console_agent_conversation(
    conversation_id: str,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentConversation:
    conversation = agent_service.get_conversation(
        actor=actor,
        conversation_id=conversation_id,
    )
    if conversation is None:
        raise AppException(
            code="AGENT_CONVERSATION_NOT_FOUND",
            message="会话不存在、已过期，或当前账号无权访问。",
            status_code=404,
        )
    return conversation


@router.post(
    "/api/ai/agent/conversations/{conversation_id}/feedback",
    response_model=ConsoleAgentFeedbackResponse,
)
def submit_console_agent_feedback(
    conversation_id: str,
    request: ConsoleAgentFeedbackRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentFeedbackResponse:
    logger.info(
        "console_agent_feedback_requested conversation_id=%s actor_id=%s rating=%s",
        conversation_id,
        actor.user_id,
        request.rating,
    )
    return agent_service.submit_feedback(
        actor=actor,
        conversation_id=conversation_id,
        request=request,
    )


@router.get("/api/ai/agent/memory")
def get_agent_memory(
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
) -> dict:
    from app.services.user_memory import UserMemoryStore

    facts = UserMemoryStore(settings=get_settings()).get_facts(actor)
    return {"facts": facts}


@router.delete("/api/ai/agent/memory")
def clear_agent_memory(
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
) -> dict:
    from app.services.user_memory import UserMemoryStore

    UserMemoryStore(settings=get_settings()).clear(actor)
    return {"cleared": True}


@router.post(
    "/api/ai/agent/conversations/{conversation_id}/human-reply",
    response_model=ConsoleAgentResponse,
)
def console_agent_human_reply(
    conversation_id: str,
    request: HumanReplyRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentResponse:
    logger.info(
        "console_agent_human_reply_requested conversation_id=%s actor_id=%s",
        conversation_id,
        actor.user_id,
    )
    return agent_service.human_reply(
        actor=actor,
        conversation_id=conversation_id,
        content=request.content,
    )


@router.get(
    "/api/ai/agent/conversations/{conversation_id}/summary",
    response_model=ConversationSummaryView,
)
def console_agent_conversation_summary(
    conversation_id: str,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConversationSummaryView:
    logger.info(
        "console_agent_summary_requested conversation_id=%s actor_id=%s",
        conversation_id,
        actor.user_id,
    )
    return agent_service.conversation_summary(
        actor=actor,
        conversation_id=conversation_id,
    )


@router.post(
    "/api/ai/agent/conversations/{conversation_id}/human-handoff",
    response_model=ConsoleAgentResponse,
)
def request_console_agent_human_handoff(
    conversation_id: str,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentResponse:
    logger.info(
        "console_agent_human_handoff_requested conversation_id=%s actor_id=%s",
        conversation_id,
        actor.user_id,
    )
    return agent_service.request_human_handoff(
        actor=actor,
        conversation_id=conversation_id,
    )


@router.post(
    "/api/ai/agent/conversations/{conversation_id}/confirmations/{confirmation_id}",
    response_model=ConsoleAgentResponse,
)
def decide_console_agent_confirmation(
    conversation_id: str,
    confirmation_id: str,
    request: ConsoleAgentConfirmationRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentResponse:
    logger.info(
        "console_agent_confirmation_requested conversation_id=%s actor_id=%s approved=%s",
        conversation_id,
        actor.user_id,
        request.approved,
    )
    return agent_service.decide_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=confirmation_id,
        approved=request.approved,
    )


@router.put(
    "/api/ai/agent/conversations/{conversation_id}/confirmations/{confirmation_id}",
    response_model=ConsoleAgentResponse,
)
def correct_console_agent_confirmation(
    conversation_id: str,
    confirmation_id: str,
    request: ConsoleAgentTicketCorrectionRequest,
    actor: ConsoleAgentActor = Depends(get_console_agent_actor),
    agent_service: ConsoleAgentService = Depends(get_console_agent_service),
) -> ConsoleAgentResponse:
    logger.info(
        "console_agent_confirmation_correction_requested conversation_id=%s actor_id=%s",
        conversation_id,
        actor.user_id,
    )
    return agent_service.correct_ticket_confirmation(
        actor=actor,
        conversation_id=conversation_id,
        confirmation_id=confirmation_id,
        ticket_fields=request.ticket_fields,
    )
