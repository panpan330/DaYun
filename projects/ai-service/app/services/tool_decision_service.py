import json
import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage
from app.schemas.tool import QueryOrderArgs, ToolDefinition
from app.schemas.tool_decision import (
    ToolCallCandidate,
    ToolDecisionResponse,
    ToolDecisionType,
)
from app.services.llm_client import create_openai_compatible_client
from app.services.llm_service import (
    extract_token_usage,
    map_openai_error_to_app_exception,
)
from app.services.message_builder import (
    build_multi_turn_messages,
    serialize_chat_messages,
)
from app.tools.tool_registry import (
    authorize_tool_call,
    list_model_callable_openai_tools,
)


logger = logging.getLogger(__name__)

TOOL_DECISION_SYSTEM_PROMPT = (
    "你是一个客服 AI 工具调用决策器。"
    "你只能决定是否需要调用后端提供的工具，不能自己假装已经查询了业务系统。"
    "当用户询问订单状态、物流、是否能创建工单，并且提供了明确订单号时，优先请求 query_order 工具。"
    "当用户没有提供订单号、不需要业务数据、只是问概念或普通问题时，直接用中文回答。"
    "不要编造订单号，不要请求未提供的工具，不要请求写入或敏感操作。"
)


def build_tool_decision_prompt(user_message: str) -> str:
    return (
        "请判断下面这条用户消息是否需要调用后端工具。\n\n"
        "判断规则：\n"
        "1. 需要真实订单状态或物流信息，并且消息里有订单号，才调用 query_order。\n"
        "2. 没有订单号时，不要调用工具，直接提醒用户提供订单号。\n"
        "3. 普通知识解释、闲聊、学习类问题，不要调用工具，直接回答。\n"
        "4. 不要猜测参数，不要编造工具结果。\n\n"
        f"用户消息：\n{user_message}"
    )


def build_tool_decision_messages(
    user_message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    messages = build_multi_turn_messages(
        build_tool_decision_prompt(user_message),
        history=history,
        system_message=TOOL_DECISION_SYSTEM_PROMPT,
    )
    return serialize_chat_messages(messages)


def _get_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def extract_first_message(completion: Any) -> Any:
    choices = _get_field(completion, "choices")
    try:
        return _get_field(choices[0], "message")
    except (IndexError, TypeError) as exc:
        raise AppException(
            code="LLM_BAD_RESPONSE",
            message="模型返回格式异常",
            status_code=502,
        ) from exc


def extract_message_tool_calls(message: Any) -> list[Any]:
    tool_calls = _get_field(message, "tool_calls")
    if tool_calls is None:
        return []
    if isinstance(tool_calls, list):
        return tool_calls
    if isinstance(tool_calls, tuple):
        return list(tool_calls)
    raise AppException(
        code="TOOL_DECISION_BAD_RESPONSE",
        message="模型返回的工具调用格式异常",
        status_code=502,
    )


def extract_direct_reply(message: Any) -> str:
    content = _get_field(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise AppException(
            code="LLM_EMPTY_RESPONSE",
            message="模型返回了空内容",
            status_code=502,
        )
    return content.strip()


def parse_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise AppException(
                code="TOOL_ARGUMENTS_INVALID_JSON",
                message="模型返回的工具参数不是合法 JSON",
                status_code=502,
            ) from exc
    else:
        raise AppException(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message="模型返回的工具参数不是合法 JSON",
            status_code=502,
        )

    if not isinstance(arguments, dict):
        raise AppException(
            code="TOOL_ARGUMENTS_INVALID_JSON",
            message="模型返回的工具参数必须是 JSON 对象",
            status_code=502,
        )
    return arguments


def validate_tool_arguments(
    definition: ToolDefinition,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if definition.name != "query_order":
        raise AppException(
            code="TOOL_NOT_ALLOWED",
            message="工具不在允许列表中，后端已拒绝执行。",
            status_code=403,
        )

    try:
        return QueryOrderArgs.model_validate(arguments).model_dump()
    except ValidationError as exc:
        raise AppException(
            code="TOOL_ARGUMENTS_VALIDATION_FAILED",
            message="模型返回的工具参数校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False),
        ) from exc


def extract_tool_decision(completion: Any) -> ToolDecisionResponse:
    message = extract_first_message(completion)
    tool_calls = extract_message_tool_calls(message)
    if not tool_calls:
        return ToolDecisionResponse(
            decision=ToolDecisionType.ANSWER_DIRECTLY,
            reply=extract_direct_reply(message),
        )

    if len(tool_calls) > 1:
        raise AppException(
            code="TOOL_DECISION_TOO_MANY_CALLS",
            message="当前阶段一次只支持一个工具调用请求。",
            status_code=502,
        )

    tool_call = tool_calls[0]
    function = _get_field(tool_call, "function")
    tool_name = _get_field(function, "name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise AppException(
            code="TOOL_DECISION_BAD_RESPONSE",
            message="模型返回的工具名称格式异常",
            status_code=502,
        )

    definition = authorize_tool_call(tool_name.strip())
    raw_arguments = _get_field(function, "arguments")
    parsed_arguments = parse_tool_call_arguments(raw_arguments)
    validated_arguments = validate_tool_arguments(definition, parsed_arguments)

    call_id = _get_field(tool_call, "id")
    if not isinstance(call_id, str):
        call_id = None

    return ToolDecisionResponse(
        decision=ToolDecisionType.CALL_TOOL,
        tool_call=ToolCallCandidate(
            name=definition.name,
            arguments=validated_arguments,
            call_id=call_id,
        ),
    )


class ToolDecisionService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

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

    def _log_success(
        self,
        elapsed_ms: float,
        completion: Any,
        decision: ToolDecisionResponse,
    ) -> None:
        usage = extract_token_usage(completion)
        tool_name = decision.tool_call.name if decision.tool_call else None
        logger.info(
            (
                "tool_decision_succeeded provider=%s model=%s elapsed_ms=%.2f "
                "decision=%s tool_name=%s prompt_tokens=%s completion_tokens=%s "
                "total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
            decision.decision.value,
            tool_name,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    def _log_failure(
        self,
        app_exception: AppException,
        elapsed_ms: float,
        *,
        exc_info: bool = False,
    ) -> None:
        logger.warning(
            (
                "tool_decision_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )

    def decide(
        self,
        user_message: str,
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> ToolDecisionResponse:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_tool_decision_messages(user_message, history=history)
        start_time = perf_counter()
        try:
            completion = self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                tools=list_model_callable_openai_tools(),
                tool_choice="auto",
            )
            decision = extract_tool_decision(completion)
        except AppException as exc:
            self._log_failure(exc, (perf_counter() - start_time) * 1000)
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(
                app_exception,
                (perf_counter() - start_time) * 1000,
                exc_info=True,
            )
            raise app_exception from exc

        self._log_success((perf_counter() - start_time) * 1000, completion, decision)
        return decision


def create_tool_decision_service(settings: Settings) -> ToolDecisionService:
    return ToolDecisionService(settings)
