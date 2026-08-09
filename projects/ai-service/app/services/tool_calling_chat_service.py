import json
import logging
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage
from app.schemas.tool import QueryOrderArgs, QueryOrderResult
from app.schemas.tool_decision import ToolCallCandidate, ToolDecisionType
from app.services.llm_client import create_openai_compatible_client
from app.services.llm_service import (
    extract_token_usage,
    map_openai_error_to_app_exception,
)
from app.services.message_builder import (
    build_multi_turn_messages,
    serialize_chat_messages,
)
from app.services.tool_decision_service import (
    extract_direct_reply,
    extract_first_message,
    extract_message_tool_calls,
    extract_tool_decision,
)
from app.tools.fake_order_tool import query_order
from app.tools.tool_registry import (
    authorize_tool_call,
    list_model_callable_openai_tools,
)
from app.tools.tool_selection import (
    build_default_tool_selection_classifier,
    select_tool_names,
)


logger = logging.getLogger(__name__)

TOOL_CHAT_SYSTEM_PROMPT = (
    "你是一个客服 AI 助手。你可以请求后端提供的只读工具，但不能假装已经查询了业务系统。"
    "当用户需要真实订单状态、物流或是否能创建工单，并且提供了明确订单号时，优先请求 query_order。"
    "当用户没有提供订单号、不需要业务数据、只是问概念或普通问题时，直接用中文回答。"
    "当后端通过 tool message 返回工具结果后，只能依据该结果生成清楚、自然的中文回答，"
    "不能编造订单信息，也不要在当前轮继续请求工具。"
)

OrderQueryExecutor = Callable[[QueryOrderArgs], QueryOrderResult]


def build_tool_chat_prompt(user_message: str) -> str:
    return (
        "请处理下面这条用户消息。\n\n"
        "处理规则：\n"
        "1. 需要真实订单状态或物流信息，并且消息里有订单号，才调用 query_order。\n"
        "2. 没有订单号时，不要调用工具，直接提醒用户提供订单号。\n"
        "3. 普通知识解释、闲聊、学习类问题，不要调用工具，直接回答。\n"
        "4. 不要猜测参数，不要编造工具结果。\n\n"
        f"用户消息：\n{user_message}"
    )


def build_tool_chat_messages(
    user_message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    messages = build_multi_turn_messages(
        build_tool_chat_prompt(user_message),
        history=history,
        system_message=TOOL_CHAT_SYSTEM_PROMPT,
    )
    return serialize_chat_messages(messages)


def require_tool_call_id(tool_call: ToolCallCandidate) -> str:
    if isinstance(tool_call.call_id, str) and tool_call.call_id.strip():
        return tool_call.call_id
    raise AppException(
        code="TOOL_CALL_ID_MISSING",
        message="模型返回的工具调用缺少关联编号。",
        status_code=502,
    )


def build_assistant_tool_call_message(
    tool_call: ToolCallCandidate,
) -> dict[str, Any]:
    call_id = require_tool_call_id(tool_call)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    }


def build_tool_result_message(
    tool_call: ToolCallCandidate,
    result: QueryOrderResult,
) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": require_tool_call_id(tool_call),
        "content": json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def build_tool_summary_messages(
    initial_messages: Sequence[dict[str, str]],
    tool_call: ToolCallCandidate,
    result: QueryOrderResult,
) -> list[dict[str, Any]]:
    return [
        *initial_messages,
        build_assistant_tool_call_message(tool_call),
        build_tool_result_message(tool_call, result),
    ]


def extract_tool_summary_reply(completion: Any) -> str:
    message = extract_first_message(completion)
    if extract_message_tool_calls(message):
        raise AppException(
            code="TOOL_SUMMARY_UNEXPECTED_TOOL_CALL",
            message="模型在工具结果返回后又请求了工具，当前阶段暂不支持继续调用。",
            status_code=502,
        )
    return extract_direct_reply(message)


class ToolCallingChatService:
    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        query_order_executor: OrderQueryExecutor | None = None,
        tool_selector: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._query_order_executor = query_order_executor
        self._tool_selector = tool_selector

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            self._client = create_openai_compatible_client(self.settings)
        except ValueError as exc:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            ) from exc
        return self._client

    def _build_model_tools(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
        """按对话动态选择暴露给模型的工具集；开关关闭时全量。"""
        if not self.settings.tool_selection_enabled:
            return list_model_callable_openai_tools()
        user_messages = [
            str(item.get("content") or "")
            for item in messages
            if item.get("role") == "user"
        ]
        current = user_messages[-1] if user_messages else ""
        history = user_messages[:-1]
        classifier = (
            getattr(self, "_tool_selector", None)
            if getattr(self, "_tool_selector", None) is not None
            else build_default_tool_selection_classifier(
                self.settings.ticket_agent_model_mode,
                self.settings,
            )
        )
        selected = select_tool_names(current, history, classifier=classifier)
        return list_model_callable_openai_tools(selected)

    def _call_model(self, messages: Sequence[dict[str, Any]]) -> Any:
        try:
            return self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=list(messages),
                tools=self._build_model_tools(messages),
                tool_choice="auto",
            )
        except AppException:
            raise
        except Exception as exc:
            raise map_openai_error_to_app_exception(exc) from exc

    def _execute_tool_call(self, tool_call: ToolCallCandidate) -> QueryOrderResult:
        tool_call_id = require_tool_call_id(tool_call)
        definition = authorize_tool_call(tool_call.name)
        if definition.name != "query_order":
            raise AppException(
                code="TOOL_NOT_ALLOWED",
                message="工具不在允许列表中，后端已拒绝执行。",
                status_code=403,
            )

        try:
            arguments = QueryOrderArgs.model_validate(tool_call.arguments)
        except ValidationError as exc:
            raise AppException(
                code="TOOL_ARGUMENTS_VALIDATION_FAILED",
                message="模型返回的工具参数校验失败，请稍后重试。",
                status_code=502,
                details=exc.errors(include_url=False),
            ) from exc

        start_time = perf_counter()
        logger.info(
            "tool_execution_started tool_name=%s tool_call_id=%s order_id=%s",
            definition.name,
            tool_call_id,
            arguments.order_id,
        )
        try:
            if self._query_order_executor is not None:
                result = self._query_order_executor(arguments)
            else:
                result = query_order(arguments, settings=self.settings)
        except AppException as exc:
            logger.warning(
                (
                    "tool_execution_failed tool_name=%s tool_call_id=%s "
                    "order_id=%s code=%s status_code=%s elapsed_ms=%.2f"
                ),
                definition.name,
                tool_call_id,
                arguments.order_id,
                exc.code,
                exc.status_code,
                (perf_counter() - start_time) * 1000,
            )
            raise

        logger.info(
            (
                "tool_execution_succeeded tool_name=%s tool_call_id=%s "
                "order_id=%s source=%s elapsed_ms=%.2f"
            ),
            definition.name,
            tool_call_id,
            result.order_id,
            result.source,
            (perf_counter() - start_time) * 1000,
        )
        return result

    def _log_success(
        self,
        elapsed_ms: float,
        *,
        model_call_count: int,
        tool_name: str | None,
        completion: Any,
    ) -> None:
        usage = extract_token_usage(completion)
        logger.info(
            (
                "tool_chat_succeeded provider=%s model=%s elapsed_ms=%.2f "
                "model_call_count=%s tool_name=%s prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
            model_call_count,
            tool_name,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    def _log_failure(self, app_exception: AppException, elapsed_ms: float) -> None:
        logger.warning(
            (
                "tool_chat_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
        )

    def generate_reply(
        self,
        user_message: str,
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> str:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        initial_messages = build_tool_chat_messages(user_message, history=history)
        start_time = perf_counter()
        try:
            first_completion = self._call_model(initial_messages)
            decision = extract_tool_decision(first_completion)
            if decision.decision == ToolDecisionType.ANSWER_DIRECTLY:
                if decision.reply is None:
                    raise AppException(
                        code="LLM_EMPTY_RESPONSE",
                        message="模型返回了空内容",
                        status_code=502,
                    )
                self._log_success(
                    (perf_counter() - start_time) * 1000,
                    model_call_count=1,
                    tool_name=None,
                    completion=first_completion,
                )
                return decision.reply

            if decision.tool_call is None:
                raise AppException(
                    code="TOOL_DECISION_BAD_RESPONSE",
                    message="模型返回的工具调用结构异常",
                    status_code=502,
                )

            require_tool_call_id(decision.tool_call)
            result = self._execute_tool_call(decision.tool_call)
            summary_messages = build_tool_summary_messages(
                initial_messages,
                decision.tool_call,
                result,
            )
            second_completion = self._call_model(summary_messages)
            reply = extract_tool_summary_reply(second_completion)
        except AppException as exc:
            self._log_failure(exc, (perf_counter() - start_time) * 1000)
            raise

        self._log_success(
            (perf_counter() - start_time) * 1000,
            model_call_count=2,
            tool_name=decision.tool_call.name,
            completion=second_completion,
        )
        return reply


def create_tool_calling_chat_service(settings: Settings) -> ToolCallingChatService:
    return ToolCallingChatService(settings)
