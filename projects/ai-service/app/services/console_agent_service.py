from dataclasses import dataclass
from threading import Lock
from collections.abc import Iterator
import json
import logging
from typing import Any, Protocol
from uuid import uuid4

import httpx
from langgraph.checkpoint.redis import RedisSaver
from redis.exceptions import RedisError

from app.agents.ticket_agent import (
    LLMTicketFields,
    TICKET_CONFIRMATION_INTERRUPT_KIND,
    TicketFields,
    build_ticket_agent_input,
    build_pending_ticket_confirmation,
    build_ticket_agent_graph_for_model_mode,
    build_ticket_agent_thread_config,
    create_java_cancel_executor,
    create_java_refund_executor,
    create_ticket_agent_model_dependencies,
    get_ticket_confirmation_interrupt_payload,
    find_missing_ticket_fields,
    resume_ticket_confirmation_interrupt,
    run_ticket_agent_in_thread,
)
from app.core.ai_security_boundary import redact_sensitive_text
from app.core.business_context import reset_business_context, set_business_context
from app.core.config import Settings
from app.core.exceptions import AppException
from app.core.trace import build_trace_headers, get_trace_id
from app.rag.filters import RagAccessScope
from app.rag.generator import RagAnswer, create_rag_answer_service
from app.rag.pipeline import enhanced_rag_answer
from app.schemas.console_agent import (
    ConsoleAgentConversation,
    ConsoleAgentConversationSummary,
    ConsoleAgentFeedbackRequest,
    ConsoleAgentFeedbackResponse,
    ConsoleAgentResponse,
    ConsoleAgentHumanHandoff,
    ConsoleAgentTicketConfirmation,
    ConsoleAgentTicketFields,
    ConversationSummaryView,
)
from app.schemas.ticket import CreatedTicket
from app.services.java_ticket_client import JavaTicketClient
from app.services.java_feedback_client import JavaFeedbackClient
from app.services.java_human_handoff_client import JavaHumanHandoffClient, JavaHumanHandoffClientError
from app.services.console_agent_conversation_store import ConsoleAgentConversationStore
from app.tools.fake_order_tool import query_order


logger = logging.getLogger(__name__)


ROLE_TO_PERMISSION_GROUPS: dict[str, list[str]] = {
    "customer": ["customer_service", "public"],
    "staff": ["customer_service", "public", "internal_staff"],
    "agent": ["customer_service", "public", "internal_staff"],
    "admin": ["customer_service", "public", "internal_staff", "admin"],
}
_DEFAULT_PERMISSION_GROUPS = ["customer_service", "public"]


def build_rag_access_scope(*, user_id: str, tenant_id: str, roles: tuple[str, ...]) -> RagAccessScope:
    """Map actor roles to the permission groups they may retrieve from the knowledge base."""
    permission_groups: list[str] = []
    for role in roles:
        for group in ROLE_TO_PERMISSION_GROUPS.get(role, _DEFAULT_PERMISSION_GROUPS):
            if group not in permission_groups:
                permission_groups.append(group)
    return RagAccessScope(user_id=user_id, tenant_id=tenant_id, permission_groups=permission_groups)


@dataclass(frozen=True)
class ConsoleAgentActor:
    user_id: str
    tenant_id: str
    roles: tuple[str, ...]


def _serialize_emotion(emotion: object) -> str | None:
    """Serialize an emotion value to its wire form ('angry'), accepting enums."""
    if emotion is None or emotion == "":
        return None
    value = getattr(emotion, "value", emotion)
    return str(value) or None


HUMAN_TAKEOVER_REPLY = "已为您转接人工客服，请稍候"


AGENT_PROGRESS_BY_NODE: dict[str, tuple[str, str]] = {
    "normalize_user_input": ("preparing", "正在准备本次请求"),
    "supervisor_route": ("analyzing", "正在分析问题类型"),
    "classify_intent": ("analyzing", "正在分析问题类型"),
    "retrieve_policy": ("knowledge_search", "正在检索知识库"),
    "decide_ticket_need": ("planning", "正在规划处理方式"),
    "query_order": ("order_lookup", "正在查询订单信息"),
    "extract_ticket_fields": ("ticket_draft", "正在整理工单信息"),
    "handle_refund_request": ("refund_draft", "正在整理退款信息"),
    "handle_cancel_request": ("cancel_draft", "正在整理取消订单信息"),
    "ask_missing_ticket_fields": ("need_details", "正在确认需要补充的信息"),
    "request_ticket_confirmation": ("confirmation", "正在准备工单确认"),
    "create_ticket": ("ticket_creation", "正在创建工单"),
    "execute_refund_request": ("refund_execution", "正在执行退款"),
    "execute_cancel_request": ("cancel_execution", "正在执行取消订单"),
    "build_direct_answer": ("answering", "正在整理回复"),
    "build_unsupported_answer": ("answering", "正在整理回复"),
    "ask_clarifying_question": ("need_details", "正在确认需要补充的信息"),
}

HUMAN_HANDOFF_BLOCKED_ORDER_ERROR_CODES = frozenset(
    {"ORDER_ACCESS_DENIED", "ORDER_NOT_FOUND", "ORDER_ID_INVALID"}
)


def build_agent_progress_event(node_name: str) -> dict[str, str] | None:
    if node_name == "__interrupt__":
        return {"stage": "waiting_confirmation", "label": "等待你的确认"}
    progress = AGENT_PROGRESS_BY_NODE.get(node_name)
    if progress is None:
        return None
    stage, label = progress
    return {"stage": stage, "label": label}


def build_agent_detail_events(
    node_name: str,
    node_update: dict,
    settings,
) -> list[dict]:
    """Emit retrieval/tool/model detail events from node updates; empty when no detail."""
    if not getattr(settings, "agent_detail_events_enabled", False):
        return []
    events: list[dict] = []
    if node_name == "retrieve_policy":
        citations = node_update.get("rag_citations") or []
        if citations:
            events.append(
                {
                    "event": "detail",
                    "data": {
                        "kind": "retrieval",
                        "kb": str(citations[0].get("source", "知识库")),
                        "hits": len(citations),
                    },
                }
            )
    elif node_name in (
        "query_order",
        "handle_refund_request",
        "handle_cancel_request",
        "create_ticket",
    ):
        order_id = node_update.get("order_query_order_id") or node_update.get("order_id")
        if order_id:
            events.append(
                {
                    "event": "detail",
                    "data": {
                        "kind": "tool",
                        "tool": node_name,
                        "order_id": str(order_id),
                    },
                }
            )
    elif node_name in (
        "build_direct_answer",
        "build_unsupported_answer",
        "build_grounded_answer",
    ):
        events.append(
            {
                "event": "detail",
                "data": {
                    "kind": "model",
                    "model": getattr(settings, "resolved_llm_model", "unknown"),
                },
            }
        )
    return events


class ConsoleAgentActorResolver(Protocol):
    def resolve(self, authorization: str | None) -> ConsoleAgentActor:
        """Resolve the caller from the Java business service authentication boundary."""


class JavaConsoleAgentActorResolver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(self, authorization: str | None) -> ConsoleAgentActor:
        if not authorization or not authorization.strip():
            raise AppException(
                code="AUTH_REQUIRED",
                message="Please sign in before using the AI customer service.",
                status_code=401,
            )

        base_url = self.settings.resolved_java_business_service_base_url
        try:
            with httpx.Client(base_url=base_url, timeout=self.settings.resolved_java_business_service_timeout_seconds) as client:
                response = client.get(
                    "/api/auth/me",
                    headers={
                        "Authorization": authorization,
                        **build_trace_headers(),
                    },
                )
        except httpx.RequestError as exc:
            raise AppException(
                code="AUTH_SERVICE_UNAVAILABLE",
                message="The account service is temporarily unavailable. Please try again later.",
                status_code=502,
            ) from exc

        if response.status_code == 401:
            raise AppException(
                code="AUTH_REQUIRED",
                message="Your sign-in has expired. Please sign in again.",
                status_code=401,
            )
        if response.status_code != 200:
            raise AppException(
                code="AUTH_SERVICE_REJECTED",
                message="The account service could not verify the current user.",
                status_code=502,
            )

        try:
            payload = response.json()
            data = payload["data"]
            user_id = str(data["user_id"]).strip()
            tenant_id = str(data["tenant_id"]).strip()
            roles = tuple(str(role).strip() for role in data["roles"] if str(role).strip())
        except (KeyError, TypeError, ValueError) as exc:
            raise AppException(
                code="AUTH_SERVICE_RESPONSE_INVALID",
                message="The account service returned an invalid user identity response.",
                status_code=502,
            ) from exc

        if not user_id or not tenant_id:
            raise AppException(
                code="AUTH_SERVICE_RESPONSE_INVALID",
                message="The account service returned an incomplete user identity response.",
                status_code=502,
            )
        return ConsoleAgentActor(user_id=user_id, tenant_id=tenant_id, roles=roles)


_TICKET_CONFIRMATION_PENDING_NEXT_NODES = frozenset(
    {"request_ticket_confirmation", "ticket_agent"}
)


def _has_pending_ticket_confirmation_next(snapshot: Any) -> bool:
    """Whether a snapshot is paused waiting for a ticket confirmation decision.

    单 Agent 图：顶层 next 直接含 "request_ticket_confirmation"。
    多 Agent 监督图：确认中断发生在 ticket worker 子图内部，顶层 next 为
    ("ticket_agent",)，两者都代表有待决工单确认。
    """
    return bool(set(snapshot.next) & _TICKET_CONFIRMATION_PENDING_NEXT_NODES)


def _pending_confirmation_fields(snapshot: Any) -> dict | None:
    """Resolve the confirmed ticket draft from a state snapshot.

    优先级：活动中的确认中断（interrupt payload）优先，顶层 ticket_fields 兜底。
    多 Agent 模式下 worker 子图完成会把 ticket_fields 写回顶层；同一会话第二张
    待确认工单中断时顶层仍是上一轮旧草稿，若先读顶层会用旧字段算出错误
    confirmation_id，导致确认流程误抛 TICKET_CONFIRMATION_MISMATCH。
    单 Agent 图中断时 interrupts payload 与顶层 ticket_fields 指向同一份草稿，
    调整优先级后行为不变。
    """
    for interrupt in getattr(snapshot, "interrupts", ()) or ():
        value = getattr(interrupt, "value", None)
        if not isinstance(value, dict):
            continue
        pending = value.get("pending_ticket_confirmation")
        if isinstance(pending, dict) and isinstance(pending.get("ticket_fields"), dict):
            return pending["ticket_fields"]
    fields = snapshot.values.get("ticket_fields")
    if isinstance(fields, dict):
        return fields
    return None


class ProductionPolicyRagService:
    """Adapter that lets the Agent reuse the production embedding, Qdrant, and rerank chain."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer_policy_question(
        self,
        query: str,
        *,
        access_scope: RagAccessScope | None = None,
        memory_context: list[str] | None = None,
    ) -> RagAnswer:
        return enhanced_rag_answer(
            query,
            settings=self.settings,
            access_scope=access_scope,
            memory_context=memory_context,
        )


def _reset_request_context(tokens: object) -> None:
    """Clear per-request business context and the cost intent label."""
    reset_business_context(tokens)
    from app.services.cost_recorder import set_cost_intent

    set_cost_intent("general")


class ConsoleAgentService:
    def __init__(
        self,
        settings: Settings,
        *,
        graph: Any | None = None,
        conversation_store: ConsoleAgentConversationStore | None = None,
        feedback_client: JavaFeedbackClient | None = None,
        conversation_sync: Any | None = None,
        java_conversation_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._graph = graph
        self._graph_lock = Lock()
        self._checkpointer_context: Any | None = None
        self._conversation_store = conversation_store
        self._feedback_client = feedback_client
        self._handoff_client: JavaHumanHandoffClient | None = None
        self._conversation_sync = conversation_sync
        self._java_conversation_client = java_conversation_client

    @property
    def graph(self) -> Any:
        if self._graph is None:
            with self._graph_lock:
                if self._graph is None:
                    self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> Any:
        ticket_creator, order_query_executor, refund_executor, cancel_executor = (
            self._build_tool_dependencies()
        )
        if self.settings.agent_multi_agent_enabled:
            from app.agents.supervisor.supervisor_graph import build_supervisor_graph

            emotion_classifier = create_ticket_agent_model_dependencies(
                mode=self.settings.ticket_agent_model_mode,
                settings=self.settings,
            )["emotion_classifier"]
            return build_supervisor_graph(
                knowledge_service=ProductionPolicyRagService(self.settings),
                order_query_executor=order_query_executor,
                ticket_creator=ticket_creator,
                refund_executor=refund_executor,
                cancel_executor=cancel_executor,
                checkpointer=self._create_redis_checkpointer(),
                interrupt_confirmation=True,
                emotion_classifier=emotion_classifier,
            )
        return build_ticket_agent_graph_for_model_mode(
            ticket_creator=ticket_creator,
            policy_rag_service=ProductionPolicyRagService(self.settings),
            order_query_executor=order_query_executor,
            refund_executor=refund_executor,
            cancel_executor=cancel_executor,
            mode=self.settings.ticket_agent_model_mode,
            settings=self.settings,
            checkpointer=self._create_redis_checkpointer(),
            interrupt_confirmation=True,
        )

    def _build_tool_dependencies(self) -> tuple[Any, Any, Any, Any]:
        if self.settings.agent_mcp_tools_enabled:
            from app.agents.mcp_tool_adapters import (
                create_mcp_cancel_executor,
                create_mcp_order_query_executor,
                create_mcp_refund_executor,
                create_mcp_ticket_creator,
            )

            return (
                create_mcp_ticket_creator(self.settings),
                create_mcp_order_query_executor(self.settings),
                create_mcp_refund_executor(self.settings),
                create_mcp_cancel_executor(self.settings),
            )
        from app.tools.fake_order_tool import query_order

        return (
            JavaTicketClient.from_settings(self.settings),
            lambda arguments: query_order(arguments, settings=self.settings),
            create_java_refund_executor(),
            create_java_cancel_executor(),
        )

    def close(self) -> None:
        if self._checkpointer_context is not None:
            self._checkpointer_context.__exit__(None, None, None)
            self._checkpointer_context = None
        if self._conversation_store is not None:
            self._conversation_store.close()
            self._conversation_store = None

    def attach_conversation_sync(self, sync: Any) -> None:
        self._conversation_sync = sync

    def _build_history_window(self, *, actor: ConsoleAgentActor, conversation_id: str) -> list[str]:
        conversation = self.conversation_store.get(actor=actor, conversation_id=conversation_id)
        if conversation is None:
            return []
        messages = [{"role": message.role, "content": message.content} for message in conversation.messages]
        from app.services.conversation_history import build_history_window

        return build_history_window(
            messages,
            max_rounds=self.settings.agent_history_max_rounds,
            max_tokens=self.settings.agent_history_max_tokens,
        )

    @property
    def conversation_store(self) -> ConsoleAgentConversationStore:
        if self._conversation_store is None:
            java_client = self._java_conversation_client
            if java_client is None:
                from app.services.java_conversation_client import JavaConversationClient

                java_client = JavaConversationClient.from_settings(self.settings)
            self._conversation_store = ConsoleAgentConversationStore(
                self.settings,
                java_client=java_client,
            )
        return self._conversation_store

    @property
    def feedback_client(self) -> JavaFeedbackClient:
        if self._feedback_client is None:
            self._feedback_client = JavaFeedbackClient.from_settings(self.settings)
        return self._feedback_client

    @property
    def handoff_client(self) -> JavaHumanHandoffClient:
        if self._handoff_client is None:
            self._handoff_client = JavaHumanHandoffClient.from_settings(self.settings)
        return self._handoff_client

    def _create_redis_checkpointer(self) -> RedisSaver:
        context = RedisSaver.from_conn_string(
            self.settings.resolved_agent_redis_url,
            ttl={
                "default_ttl": self.settings.agent_checkpoint_ttl_minutes,
                "refresh_on_read": True,
            },
            checkpoint_prefix=f"{self.settings.resolved_agent_checkpoint_key_prefix}:checkpoint",
            checkpoint_write_prefix=(
                f"{self.settings.resolved_agent_checkpoint_key_prefix}:checkpoint-write"
            ),
        )
        try:
            checkpointer = context.__enter__()
            checkpointer.setup()
        except (RedisError, OSError, ValueError) as exc:
            context.__exit__(type(exc), exc, exc.__traceback__)
            raise AppException(
                code="AGENT_STATE_STORE_UNAVAILABLE",
                message="The AI conversation state service is temporarily unavailable.",
                status_code=503,
            ) from exc

        self._checkpointer_context = context
        return checkpointer

    def _active_human_handoff(self, conversation_id: str) -> dict | None:
        """Active (in_progress) handoff for the conversation; Java failure degrades to None."""
        try:
            handoff = self.handoff_client.get_active_handoff(conversation_id)
        except JavaHumanHandoffClientError as exc:
            logger.warning(
                "console_agent_handoff_lookup_failed conversation_id=%s code=%s",
                conversation_id,
                exc.code,
            )
            return None
        if handoff is None or handoff.get("status") != "in_progress":
            return None
        return handoff

    def reply(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        message: str,
    ) -> ConsoleAgentResponse:
        thread_id = self._thread_id(actor, conversation_id)
        self._reject_if_confirmation_is_pending(thread_id)
        active_handoff = self._active_human_handoff(conversation_id)
        if active_handoff is not None:
            trace_id = uuid4().hex
            response = ConsoleAgentResponse(
                reply=HUMAN_TAKEOVER_REPLY,
                conversation_id=conversation_id,
                trace_id=trace_id,
                route="human_handoff",
            )
            self.conversation_store.append_exchange(
                actor=actor,
                conversation_id=conversation_id,
                user_message=message,
                response=response,
            )
            return response
        from app.agents.langsmith_tracing import build_ticket_agent_langsmith_trace_context
        from app.agents.tracing_spans import start_agent_span

        trace_context = build_ticket_agent_langsmith_trace_context(
            {"user_message": message},
            operation="console_agent_reply",
            thread_id=thread_id,
            actor_id=actor.user_id,
            extra_tags=["console-agent"],
        )
        graph_config = trace_context.to_langgraph_config()
        with start_agent_span(
            intent=None,
            thread_id=thread_id,
            conversation_id=conversation_id,
        ):
            tokens = set_business_context(user_id=actor.user_id, tenant_id=actor.tenant_id)
            try:
                history = self._build_history_window(actor=actor, conversation_id=conversation_id)
                state = run_ticket_agent_in_thread(
                    self.graph,
                    message,
                    thread_id=thread_id,
                    actor_id=actor.user_id,
                    config=graph_config,
                    history=history,
                    actor_roles=actor.roles,
                    actor_tenant_id=actor.tenant_id,
                    memory_context=self._load_user_memory(actor),
                )
            finally:
                _reset_request_context(tokens)
        response = self._to_response(state, conversation_id=conversation_id)
        self._record_exchange(
            actor=actor,
            conversation_id=conversation_id,
            user_message=message,
            response=response,
        )
        self._maybe_extract_user_memory(actor, conversation_id, message)
        return response

    def stream_reply(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        message: str,
        trace_id: str,
    ) -> Iterator[dict[str, Any]]:
        yield {
            "event": "start",
            "data": {"trace_id": trace_id, "conversation_id": conversation_id},
        }
        thread_id = self._thread_id(actor, conversation_id)
        active_handoff = self._active_human_handoff(conversation_id)
        if active_handoff is not None:
            self.conversation_store.append_exchange(
                actor=actor,
                conversation_id=conversation_id,
                user_message=message,
                response=ConsoleAgentResponse(
                    reply=HUMAN_TAKEOVER_REPLY,
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                    route="human_handoff",
                ),
            )
            yield {
                "event": "reply",
                "data": {"content": HUMAN_TAKEOVER_REPLY, "trace_id": trace_id},
            }
            yield {
                "event": "done",
                "data": {"trace_id": trace_id, "conversation_id": conversation_id},
            }
            return
        from app.agents.langsmith_tracing import build_ticket_agent_langsmith_trace_context
        from app.agents.tracing_spans import set_span_status_error, start_agent_span

        trace_context = build_ticket_agent_langsmith_trace_context(
            {"user_message": message},
            operation="console_agent_stream",
            thread_id=thread_id,
            actor_id=actor.user_id,
            extra_tags=["console-agent"],
        )
        graph_config = trace_context.to_langgraph_config()
        context_tokens = None
        interrupt_payload: Any | None = None
        with start_agent_span(
            intent=None,
            thread_id=thread_id,
            conversation_id=conversation_id,
        ):
            try:
                self._reject_if_confirmation_is_pending(thread_id)
                context_tokens = set_business_context(
                    user_id=actor.user_id,
                    tenant_id=actor.tenant_id,
                )
                for update in self.graph.stream(
                    build_ticket_agent_input(
                        message, memory_context=self._load_user_memory(actor)
                    )
                    | {
                        "ticket_actor_id": actor.user_id,
                        "actor_roles": actor.roles,
                        "actor_tenant_id": actor.tenant_id,
                    },
                    config={**build_ticket_agent_thread_config(thread_id), **graph_config},
                    stream_mode="updates",
                ):
                    for node_name, node_update in update.items():
                        if node_name == "__interrupt__":
                            interrupt_payload = node_update
                        progress_event = build_agent_progress_event(node_name)
                        if progress_event is not None:
                            yield {"event": "stage", "data": progress_event}
                        for detail_event in build_agent_detail_events(
                            node_name, node_update, self.settings
                        ):
                            yield detail_event

                snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
                state = dict(snapshot.values)
                if interrupt_payload is not None:
                    state["__interrupt__"] = interrupt_payload
                response = self._to_response(
                    state,
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                )
            except AppException as exc:
                set_span_status_error()
                yield {
                    "event": "error",
                    "data": {
                        "code": exc.code,
                        "message": redact_sensitive_text(exc.message),
                        "trace_id": trace_id,
                    },
                }
                return
            except Exception:
                set_span_status_error()
                yield {
                    "event": "error",
                    "data": {
                        "code": "AGENT_STREAM_FAILED",
                        "message": "The AI customer service request could not be completed.",
                        "trace_id": trace_id,
                    },
                }
                return
            finally:
                if context_tokens is not None:
                    _reset_request_context(context_tokens)

        self._record_exchange(
            actor=actor,
            conversation_id=conversation_id,
            user_message=message,
            response=response,
        )
        self._maybe_extract_user_memory(actor, conversation_id, message)
        yield {"event": "result", "data": response.model_dump(mode="json")}
        yield {"event": "done", "data": {"trace_id": trace_id}}

    def decide_ticket_confirmation(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        confirmation_id: str,
        approved: bool,
    ) -> ConsoleAgentResponse:
        thread_id = self._thread_id(actor, conversation_id)
        snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
        if not _has_pending_ticket_confirmation_next(snapshot):
            raise AppException(
                code="TICKET_CONFIRMATION_NOT_FOUND",
                message="There is no pending ticket confirmation for this conversation.",
                status_code=409,
            )

        fields = _pending_confirmation_fields(snapshot)
        if fields is None:
            raise AppException(
                code="TICKET_CONFIRMATION_NOT_FOUND",
                message="The pending ticket confirmation is no longer available.",
                status_code=409,
            )
        expected_confirmation_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
        if confirmation_id.strip() != expected_confirmation_id:
            raise AppException(
                code="TICKET_CONFIRMATION_MISMATCH",
                message="The confirmation does not belong to this conversation.",
                status_code=409,
            )

        # The reliable discriminator lives in the confirmation interrupt
        # payload (written by the worker graph where refund_request_active /
        # cancel_request_active are visible): the top-level supervisor
        # snapshot.values never receives the worker flag, and the draft fields
        # alone are identical for both paths.
        is_refund_execution = self._snapshot_confirmation_is_refund_execution(
            snapshot
        )
        is_cancel_execution = self._snapshot_confirmation_is_cancel_execution(
            snapshot
        )

        tokens = set_business_context(user_id=actor.user_id, tenant_id=actor.tenant_id)
        try:
            if approved and self.settings.agent_mcp_tools_enabled:
                # Only the MCP path needs the confirmation pre-registered in the
                # shared store: the standalone MCP server re-checks it before
                # calling Java. The direct-Java path is idempotency-keyed at the
                # Java service instead, so registering here would be a dead write.
                from app.agents.mcp_tool_adapters import register_ticket_confirmation

                register_ticket_confirmation(
                    actor_id=actor.user_id,
                    fields=fields,
                    settings=self.settings,
                    is_refund_execution=is_refund_execution,
                    is_cancel_execution=is_cancel_execution,
                )
            state = resume_ticket_confirmation_interrupt(
                self.graph,
                thread_id=thread_id,
                approved=approved,
                actor_id=actor.user_id,
            )
        finally:
            _reset_request_context(tokens)
        response = self._to_response(state, conversation_id=conversation_id)
        if is_cancel_execution:
            decision_copy = "确认取消订单" if approved else "取消取消"
        elif is_refund_execution:
            decision_copy = "确认退款" if approved else "取消退款"
        else:
            decision_copy = "确认创建工单" if approved else "取消创建工单"
        self._record_exchange(
            actor=actor,
            conversation_id=conversation_id,
            user_message=decision_copy,
            response=response,
        )
        return response

    def correct_ticket_confirmation(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        confirmation_id: str,
        ticket_fields: ConsoleAgentTicketFields,
    ) -> ConsoleAgentResponse:
        thread_id = self._thread_id(actor, conversation_id)
        self._require_pending_confirmation(thread_id, confirmation_id)
        snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
        is_refund_execution = self._snapshot_confirmation_is_refund_execution(
            snapshot
        )
        is_cancel_execution = self._snapshot_confirmation_is_cancel_execution(
            snapshot
        )
        corrected_fields = self._validate_corrected_ticket_fields(ticket_fields)
        tokens = set_business_context(user_id=actor.user_id, tenant_id=actor.tenant_id)
        try:
            state = resume_ticket_confirmation_interrupt(
                self.graph,
                thread_id=thread_id,
                approved=False,
                actor_id=actor.user_id,
                corrected_fields=corrected_fields,
            )
        finally:
            _reset_request_context(tokens)
        response = self._to_response(state, conversation_id=conversation_id)
        if is_cancel_execution:
            correction_copy = "修改取消信息并重新确认"
        elif is_refund_execution:
            correction_copy = "修改退款信息并重新确认"
        else:
            correction_copy = "修改工单草稿并重新确认"
        self._record_exchange(
            actor=actor,
            conversation_id=conversation_id,
            user_message=correction_copy,
            response=response,
        )
        return response

    def human_reply(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        content: str,
    ) -> ConsoleAgentResponse:
        active = self._active_human_handoff(conversation_id)
        if active is None:
            raise AppException(
                code="HANDOFF_NOT_ACTIVE",
                message="该会话未处于人工接管状态。",
                status_code=409,
            )
        assigned = active.get("assigned_agent")
        is_supervisor = "supervisor" in actor.roles
        if assigned and assigned != actor.user_id and not is_supervisor:
            raise AppException(
                code="HANDOFF_NOT_ASSIGNED",
                message="请先接手该会话再回复。",
                status_code=403,
            )
        # 会话存储按 actor 隔离：坐席回复必须写入客户（被接管人）的会话，客户才能在 history 看到
        customer_actor = ConsoleAgentActor(
            user_id=active.get("user_id") or actor.user_id,
            tenant_id=actor.tenant_id,
            roles=("customer",),
        )
        self.conversation_store.append_human_reply(
            actor=customer_actor,
            conversation_id=conversation_id,
            content=content,
        )
        return ConsoleAgentResponse(
            reply=content,
            conversation_id=conversation_id,
            trace_id=uuid4().hex,
            route="human_handoff",
        )

    def conversation_summary(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
    ) -> ConversationSummaryView:
        active = self._active_human_handoff(conversation_id)
        if active is None:
            raise AppException(
                code="HANDOFF_NOT_ACTIVE",
                message="该会话未处于人工接管状态。",
                status_code=409,
            )
        assigned = active.get("assigned_agent")
        is_supervisor = "supervisor" in actor.roles
        if assigned and assigned != actor.user_id and not is_supervisor:
            raise AppException(
                code="HANDOFF_NOT_ASSIGNED",
                message="请先接手该会话再查看摘要。",
                status_code=403,
            )
        customer_actor = ConsoleAgentActor(
            user_id=active.get("user_id") or actor.user_id,
            tenant_id=actor.tenant_id,
            roles=("customer",),
        )
        conversation = self.conversation_store.get(
            actor=customer_actor,
            conversation_id=conversation_id,
        )
        messages = conversation.messages if conversation is not None else []
        text_messages = [m for m in messages if m.role in ("user", "assistant", "human_agent")]
        if not text_messages:
            return ConversationSummaryView(summary="暂无会话消息", source="rule")
        if self.settings.ticket_agent_model_mode == "real_llm":
            try:
                return ConversationSummaryView(
                    summary=self._llm_conversation_summary(text_messages),
                    source="llm",
                )
            except AppException:
                logger.warning(
                    "console_agent_summary_llm_fallback conversation_id=%s",
                    conversation_id,
                )
        return ConversationSummaryView(
            summary=self._rule_conversation_summary(text_messages),
            source="rule",
        )

    def _llm_conversation_summary(self, messages: list[Any]) -> str:
        from app.schemas.chat import ChatMessage, ChatMessageRole
        from app.services.llm_service import create_llm_chat_service

        history = [
            ChatMessage(
                role=(
                    ChatMessageRole.USER
                    if m.role == "user"
                    else ChatMessageRole.ASSISTANT
                ),
                content=m.content[:3900],
            )
            for m in messages[-20:]
        ]
        service = create_llm_chat_service(self.settings)
        return service.generate_reply(
            "请用 3-5 句中文总结这段客服会话：客户的诉求、已完成或正在进行的处理、需要人工跟进的事项。",
            history=history,
        )

    def _rule_conversation_summary(self, messages: list[Any]) -> str:
        first_user = next((m.content for m in messages if m.role == "user"), "")
        lines = [f"诉求：{first_user[:100]}"] if first_user else []
        for m in messages[-3:]:
            label = {"user": "客户", "human_agent": "人工", "assistant": "AI"}.get(m.role, m.role)
            lines.append(f"{label}：{m.content[:100]}")
        return "\n".join(lines)

    def request_human_handoff(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
    ) -> ConsoleAgentResponse:
        thread_id = self._thread_id(actor, conversation_id)
        self._reject_if_confirmation_is_pending(thread_id)
        snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
        state = dict(snapshot.values)
        handoff = self._human_handoff_from_state(state)
        if handoff is None:
            raise AppException(
                code="HUMAN_HANDOFF_NOT_AVAILABLE",
                message="当前会话暂不需要转交人工客服处理。",
                status_code=409,
            )

        order_hint = f"订单 {handoff.related_order_id} 的" if handoff.related_order_id else ""
        tokens = set_business_context(user_id=actor.user_id, tenant_id=actor.tenant_id)
        try:
            handoff_state = run_ticket_agent_in_thread(
                self.graph,
                f"请将{order_hint}问题转交人工客服处理。",
                thread_id=thread_id,
                actor_id=actor.user_id,
            )
        finally:
            _reset_request_context(tokens)
        response = self._to_response(handoff_state, conversation_id=conversation_id)
        self._record_exchange(
            actor=actor,
            conversation_id=conversation_id,
            user_message="请求转人工客服处理",
            response=response,
        )
        return response

    def list_conversations(
        self,
        *,
        actor: ConsoleAgentActor,
        limit: int,
    ) -> list[ConsoleAgentConversationSummary]:
        try:
            return self.conversation_store.list_recent(actor=actor, limit=limit)
        except RedisError as exc:
            raise AppException(
                code="AGENT_CONVERSATION_STORE_UNAVAILABLE",
                message="会话记录服务暂时不可用，请稍后重试。",
                status_code=503,
            ) from exc

    def get_conversation(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
    ) -> ConsoleAgentConversation | None:
        try:
            return self.conversation_store.get(actor=actor, conversation_id=conversation_id)
        except RedisError as exc:
            raise AppException(
                code="AGENT_CONVERSATION_STORE_UNAVAILABLE",
                message="会话记录服务暂时不可用，请稍后重试。",
                status_code=503,
            ) from exc

    def submit_feedback(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        request: ConsoleAgentFeedbackRequest,
    ) -> ConsoleAgentFeedbackResponse:
        conversation = self.get_conversation(actor=actor, conversation_id=conversation_id)
        if conversation is None:
            raise AppException(
                code="AGENT_CONVERSATION_NOT_FOUND",
                message="The conversation does not exist, has expired, or is not available to this account.",
                status_code=404,
            )
        target = next(
            (
                item
                for item in reversed(conversation.messages)
                if item.role == "assistant" and item.trace_id == request.trace_id
            ),
            None,
        )
        if target is None:
            raise AppException(
                code="AGENT_RESPONSE_NOT_FOUND",
                message="The selected AI response does not belong to this conversation.",
                status_code=404,
            )
        target_index = conversation.messages.index(target)
        user_message = next(
            (
                item.content
                for item in reversed(conversation.messages[:target_index])
                if item.role == "user"
            ),
            None,
        )

        tokens = set_business_context(user_id=actor.user_id, tenant_id=actor.tenant_id)
        try:
            receipt = self.feedback_client.submit(
                conversation_id=conversation_id,
                trace_id=request.trace_id,
                rating=request.rating,
                reason=request.reason,
                agent_route=target.route or self._route_for_feedback(target),
                citation_count=len(target.citations),
                human_handoff_suggested=target.human_handoff is not None,
                user_message_excerpt=user_message,
                assistant_answer_excerpt=target.content,
                citation_summary_json=json.dumps(
                    [
                        {"source": citation.source, "title": citation.title, "chunk_id": citation.chunk_id}
                        for citation in target.citations[:10]
                    ],
                    ensure_ascii=False,
                ),
            )
        finally:
            _reset_request_context(tokens)
        try:
            self.conversation_store.set_assistant_feedback(
                actor=actor,
                conversation_id=conversation_id,
                trace_id=request.trace_id,
                rating=receipt.rating,
                reason=receipt.reason,
            )
        except RedisError:
            logger.warning(
                "console_agent_feedback_transcript_update_failed conversation_id=%s actor_id=%s",
                conversation_id,
                actor.user_id,
            )
        return ConsoleAgentFeedbackResponse(
            feedback_id=receipt.feedback_id,
            rating=receipt.rating,
            reason=receipt.reason,
        )

    @staticmethod
    def _route_for_feedback(message: Any) -> str:
        if message.pending_ticket_confirmation is not None:
            return "ticket_confirmation"
        if message.created_ticket is not None:
            return "ticket_creation"
        if message.human_handoff is not None:
            return "human_handoff"
        if message.citations:
            return "policy_rag"
        return "agent"

    def _require_pending_confirmation(
        self,
        thread_id: str,
        confirmation_id: str,
    ) -> None:
        snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
        if not _has_pending_ticket_confirmation_next(snapshot):
            raise AppException(
                code="TICKET_CONFIRMATION_NOT_FOUND",
                message="There is no pending ticket confirmation for this conversation.",
                status_code=409,
            )

        fields = _pending_confirmation_fields(snapshot)
        if fields is None:
            raise AppException(
                code="TICKET_CONFIRMATION_NOT_FOUND",
                message="The pending ticket confirmation is no longer available.",
                status_code=409,
            )
        expected_confirmation_id = build_pending_ticket_confirmation(fields)["confirmation_id"]
        if confirmation_id.strip() != expected_confirmation_id:
            raise AppException(
                code="TICKET_CONFIRMATION_MISMATCH",
                message="The confirmation does not belong to this conversation.",
                status_code=409,
            )

    def _validate_corrected_ticket_fields(
        self,
        ticket_fields: ConsoleAgentTicketFields,
    ) -> TicketFields:
        fields = LLMTicketFields.model_validate(ticket_fields.model_dump()).model_dump()
        missing_fields = find_missing_ticket_fields(fields)
        if missing_fields:
            raise AppException(
                code="TICKET_FIELDS_INCOMPLETE",
                message="The corrected ticket fields are incomplete.",
                status_code=422,
            )
        return fields

    def _reject_if_confirmation_is_pending(self, thread_id: str) -> None:
        snapshot = self.graph.get_state(build_ticket_agent_thread_config(thread_id))
        if _has_pending_ticket_confirmation_next(snapshot):
            raise AppException(
                code="TICKET_CONFIRMATION_PENDING",
                message="Please confirm or cancel the pending ticket before sending a new message.",
                status_code=409,
            )

    def _thread_id(self, actor: ConsoleAgentActor, conversation_id: str) -> str:
        return f"console-{actor.tenant_id}-{actor.user_id}-{conversation_id}"

    def _to_response(
        self,
        state: dict[str, Any],
        *,
        conversation_id: str,
        trace_id: str | None = None,
    ) -> ConsoleAgentResponse:
        pending_confirmation = self._pending_confirmation_from_state(state)
        reply = (
            self._pending_confirmation_message(state, pending_confirmation)
            if pending_confirmation is not None
            else str(state.get("final_answer") or "The AI service did not return a usable reply.")
        )
        created_ticket = self._created_ticket_from_state(state)
        route = str(state.get("intent") or ("ticket_confirmation" if pending_confirmation else "agent"))
        citations = state.get("rag_citations") if isinstance(state.get("rag_citations"), list) else []
        suggestions = state.get("rag_suggestions") if isinstance(state.get("rag_suggestions"), list) else []
        human_handoff = (
            None
            if pending_confirmation is not None
            else (
                self._emotion_handoff_from_state(state)
                or self._human_handoff_from_state(state)
            )
        )
        return ConsoleAgentResponse(
            reply=redact_sensitive_text(reply),
            conversation_id=conversation_id,
            trace_id=trace_id or get_trace_id(),
            route=route,
            citations=citations,
            suggestions=[str(item) for item in suggestions],
            pending_ticket_confirmation=pending_confirmation,
            created_ticket=created_ticket,
            human_handoff=human_handoff,
            emotion=_serialize_emotion(state.get("emotion")),
        )

    def _emotion_handoff_from_state(
        self,
        state: dict[str, Any],
    ) -> ConsoleAgentHumanHandoff | None:
        if not state.get("emotion_handoff_requested"):
            return None
        emotion = state.get("emotion")
        reason = state.get("emotion_reason") or ""
        return ConsoleAgentHumanHandoff(
            reason=f"检测到强烈情绪（{emotion}）：{reason}，建议由人工客服继续跟进。",
            related_order_id=None,
        )

    def _human_handoff_from_state(
        self,
        state: dict[str, Any],
    ) -> ConsoleAgentHumanHandoff | None:
        if (
            state.get("order_query_status") != "failed"
            or state.get("order_query_error_action") != "contact_human_support"
            or state.get("order_query_error_code") in HUMAN_HANDOFF_BLOCKED_ORDER_ERROR_CODES
        ):
            return None
        order_id = state.get("order_query_order_id")
        return ConsoleAgentHumanHandoff(
            reason="订单信息暂时无法可靠处理，建议由人工客服继续跟进。",
            related_order_id=str(order_id) if isinstance(order_id, str) and order_id.strip() else None,
        )

    def _load_user_memory(self, actor: ConsoleAgentActor) -> list[str] | None:
        if not self.settings.agent_memory_enabled:
            return None
        try:
            from app.services.user_memory import UserMemoryStore

            facts = UserMemoryStore(settings=self.settings).get_facts(actor)
            return [item["fact"] for item in facts] or None
        except Exception:
            return None

    def _maybe_extract_user_memory(
        self,
        actor: ConsoleAgentActor,
        conversation_id: str,
        user_message: str,
    ) -> None:
        if not self.settings.agent_memory_enabled:
            return
        try:
            from app.services.user_memory import (
                UserMemoryStore,
                extract_user_facts,
                extract_user_facts_llm,
            )

            store = UserMemoryStore(
                ttl_seconds=self.settings.agent_memory_ttl_minutes * 60,
                settings=self.settings,
            )
            facts = extract_user_facts(user_message, conversation_id)
            facts += extract_user_facts_llm(
                user_message,
                conversation_id,
                self.settings,
            )
            if facts:
                store.store_facts(actor, facts)
        except Exception:
            logger.warning(
                "console_agent_user_memory_extract_failed conversation_id=%s",
                conversation_id,
            )

    def _record_exchange(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        user_message: str,
        response: ConsoleAgentResponse,
    ) -> None:
        try:
            self.conversation_store.append_exchange(
                actor=actor,
                conversation_id=conversation_id,
                user_message=user_message,
                response=response,
            )
            if self._conversation_sync is not None:
                self._conversation_sync.mark_pending(
                    tenant_id=actor.tenant_id,
                    user_id=actor.user_id,
                    conversation_id=conversation_id,
                )
        except RedisError:
            logger.warning(
                "console_agent_conversation_persist_failed conversation_id=%s actor_id=%s",
                conversation_id,
                actor.user_id,
            )
        self._maybe_enqueue_handoff(
            actor=actor,
            conversation_id=conversation_id,
            response=response,
        )

    def _maybe_enqueue_handoff(
        self,
        *,
        actor: ConsoleAgentActor,
        conversation_id: str,
        response: ConsoleAgentResponse,
    ) -> None:
        """Enqueue a human handoff into the Java queue (best effort, non-blocking)."""
        handoff = response.human_handoff
        if handoff is None:
            return
        try:
            self.handoff_client.create_handoff(
                conversation_id=conversation_id,
                user_id=actor.user_id,
                tenant_id=actor.tenant_id,
                reason=handoff.reason,
                related_order_id=handoff.related_order_id,
                emotion=response.emotion,
            )
        except Exception:
            logger.warning(
                "console_agent_handoff_enqueue_failed conversation_id=%s actor_id=%s",
                conversation_id,
                actor.user_id,
            )

    def _snapshot_confirmation_is_refund_execution(self, snapshot: Any) -> bool:
        """Whether the pending confirmation interrupt is a refund execution.

        The flag is written into the interrupt payload by the worker graph
        (where refund_request_active is visible).  The top-level supervisor
        snapshot.values never carries the worker flag, so the interrupt payload
        is the only reliable source for both single- and multi-agent graphs.
        """
        for interrupt in getattr(snapshot, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if (
                isinstance(value, dict)
                and value.get("kind") == TICKET_CONFIRMATION_INTERRUPT_KIND
            ):
                return value.get("is_refund_execution") is True
        return False

    def _snapshot_confirmation_is_cancel_execution(self, snapshot: Any) -> bool:
        """Whether the pending confirmation interrupt is a cancel execution.

        The flag is written into the interrupt payload by the worker graph
        (where cancel_request_active is visible).  The top-level supervisor
        snapshot.values never carries the worker flag, so the interrupt payload
        is the only reliable source for both single- and multi-agent graphs.
        """
        for interrupt in getattr(snapshot, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if (
                isinstance(value, dict)
                and value.get("kind") == TICKET_CONFIRMATION_INTERRUPT_KIND
            ):
                return value.get("is_cancel_execution") is True
        return False

    def _pending_confirmation_from_state(
        self,
        state: dict[str, Any],
    ) -> ConsoleAgentTicketConfirmation | None:
        # The graph keeps the original draft after a decision for audit and
        # idempotency.  Only an active interrupt represents a confirmation the
        # user can still act on.
        if not state.get("__interrupt__"):
            return None

        interrupt_payload = get_ticket_confirmation_interrupt_payload(state)
        pending = interrupt_payload.get("pending_ticket_confirmation")
        if not isinstance(pending, dict):
            return None
        return ConsoleAgentTicketConfirmation.model_validate(
            {
                "confirmation_id": pending.get("confirmation_id"),
                "title": pending.get("title"),
                "summary": pending.get("summary"),
                "is_refund_execution": (
                    interrupt_payload.get("is_refund_execution") is True
                ),
                "is_cancel_execution": (
                    interrupt_payload.get("is_cancel_execution") is True
                ),
                "ticket_fields": pending.get("ticket_fields"),
            }
        )

    def _pending_confirmation_message(
        self,
        state: dict[str, Any],
        pending_confirmation: ConsoleAgentTicketConfirmation,
    ) -> str:
        interrupt_payload = None
        if state.get("__interrupt__"):
            interrupt_payload = get_ticket_confirmation_interrupt_payload(state)
        message = (
            interrupt_payload.get("message")
            if isinstance(interrupt_payload, dict)
            else state.get("ticket_confirmation_message")
        )
        return str(message or pending_confirmation.summary)

    def _created_ticket_from_state(self, state: dict[str, Any]) -> CreatedTicket | None:
        created_ticket = state.get("created_ticket")
        if not isinstance(created_ticket, dict):
            return None
        return CreatedTicket.model_validate(created_ticket)
