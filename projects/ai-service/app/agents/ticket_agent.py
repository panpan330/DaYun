from collections.abc import Callable
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from operator import add
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Protocol
from typing_extensions import TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
)

from app.agents.checkpoint_store import (
    FileTicketAgentCheckpointStore,
    TicketAgentCheckpointSnapshot,
)
from app.agents.emotion_classifier import (
    CustomerEmotion,
    CustomerEmotionClassifier,
    EMOTION_HANDOFF_EMOTIONS,
    FakeLLMEmotionClassifier,
    LLMEmotionClassifier,
    RuleBasedEmotionClassifier,
)
from app.agents.thread_lifecycle import normalize_ticket_agent_thread_id
from app.core.config import Settings, TicketAgentModelMode, get_settings
from app.core.exceptions import AppException
from app.core.trace import get_trace_id
from app.rag.documents import RetrievedChunk
from app.rag.generator import RagAnswer, build_grounded_rag_answer, build_no_context_rag_answer
from app.schemas.cancel import CancelOrderArgs
from app.schemas.refund import RefundOrderArgs
from app.schemas.ticket import (
    CreateTicketArgs,
    CreatedTicket,
    TicketCategory,
    TicketPriority,
)
from app.schemas.tool import QueryOrderArgs, QueryOrderResult, ToolDefinition
from app.services.java_ticket_client import JavaTicketClient
from app.services.llm_client import create_openai_compatible_client
from app.services.llm_service import (
    extract_first_reply,
    extract_token_usage,
    map_openai_error_to_app_exception,
)
from app.tools.fake_order_tool import query_order as run_query_order_tool
from app.tools.tool_registry import (
    CANCEL_ORDER_TOOL_NAME,
    REFUND_ORDER_TOOL_NAME,
    authorize_tool_call,
    get_tool_definition,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


logger = logging.getLogger(__name__)

from app.agents.ticket_agent_pkg.constants import (
    POLICY_KEYWORDS,
    MISSING_TICKET_FIELD_QUESTIONS,
    ORDER_REQUIRED_ISSUE_TYPES,
    ORDER_STATUS_LABELS,
    PAYMENT_STATUS_LABELS,
    TICKET_ISSUE_TYPE_LABELS,
    TICKET_ISSUE_TYPE_TO_CATEGORY,
    TICKET_URGENCY_LABELS,
    TICKET_URGENCY_TO_PRIORITY,
    ORDER_KEYWORDS,
    TICKET_KEYWORDS,
    SMALLTALK_KEYWORDS,
    UNSUPPORTED_KEYWORDS,
    CANCEL_ACTION_PHRASES,
    CANCEL_KEYWORDS,
    CANCEL_WITH_ORDER_PATTERN,
    CANCEL_QUERY_WORDS,
    REFUND_ACTION_PHRASES,
    REFUND_KEYWORDS,
    REFUND_WITH_ORDER_PATTERN,
    REFUND_QUERY_WORDS,
    UNCLEAR_MESSAGES,
    ORDER_ID_PATTERN,
    FALLBACK_ORDER_ID_PATTERN,
    REFUND_ISSUE_KEYWORDS,
    LOGISTICS_ISSUE_KEYWORDS,
    COMPLAINT_ISSUE_KEYWORDS,
    HIGH_URGENCY_KEYWORDS,
    DEFAULT_TICKET_ACTOR_ID,
    CREATE_TICKET_TOOL_NAME,
    TICKET_CONFIRMATION_NOT_FOUND_MESSAGE,
    TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND_MESSAGE,
    TICKET_CONFIRMATION_REJECTED_MESSAGE,
    TICKET_CONFIRMATION_INTERRUPT_KIND,
    REFUND_CONFIRMATION_REJECTED_MESSAGE,
    REFUND_FIELDS_NOT_FOUND_MESSAGE,
    REFUND_CONFIRMATION_REQUIRED_MESSAGE,
    REFUND_UNEXPECTED_ERROR_CODE,
    REFUND_UNEXPECTED_ERROR_MESSAGE,
    REFUND_REASON_MAX_LENGTH,
    CANCEL_CONFIRMATION_REJECTED_MESSAGE,
    CANCEL_FIELDS_NOT_FOUND_MESSAGE,
    CANCEL_CONFIRMATION_REQUIRED_MESSAGE,
    CANCEL_UNEXPECTED_ERROR_CODE,
    CANCEL_UNEXPECTED_ERROR_MESSAGE,
    CANCEL_REASON_MAX_LENGTH,
    TICKET_AGENT_FALLBACK_ERROR_CODE,
    TICKET_AGENT_FALLBACK_MESSAGE,
    TICKET_ORDER_QUERY_MISSING_ORDER_ID_MESSAGE,
    TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_ERROR_CODE,
    TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_MESSAGE,
    TICKET_ORDER_QUERY_UNEXPECTED_ERROR_CODE,
    TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE,
    TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE,
    TICKET_ORDER_QUERY_NOT_FOUND_CODES,
    TICKET_ORDER_QUERY_TIMEOUT_CODES,
    TICKET_ORDER_QUERY_UPSTREAM_ERROR_CODES,
    TICKET_ORDER_QUERY_RESULT_VALIDATION_FAILED_CODES,
    TICKET_ORDER_QUERY_TOOL_ERROR_CODES,
    TICKET_CREATION_UNEXPECTED_ERROR_CODE,
    TICKET_CREATION_UNEXPECTED_ERROR_MESSAGE,
    TICKET_THREAD_ID_INVALID_ERROR_CODE,
    TICKET_AGENT_LOG_VALUE_EMPTY,
    TICKET_AGENT_MODEL_EMPTY_RESPONSE_CODES,
    TICKET_AGENT_MODEL_SCHEMA_VALIDATION_FAILED_CODES,
    TICKET_AGENT_MODEL_TRANSIENT_PROVIDER_ERROR_CODES,
    TICKET_AGENT_MODEL_CONFIGURATION_ERROR_CODES,
)
from app.agents.ticket_agent_pkg.state import (
    CancelExecutor,
    CancelStatus,
    LLMTicketFields,
    LLMTicketIntentClassification,
    PendingTicketConfirmation,
    PolicyRagService,
    RefundExecutor,
    RefundStatus,
    TICKET_AGENT_CONFIRMATION_ROUTES,
    TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    TICKET_AGENT_FIXED_EDGES,
    TICKET_AGENT_INTENT_ROUTES,
    TICKET_AGENT_PROMPTS,
    TICKET_AGENT_TICKET_NEED_ROUTES,
    TICKET_FIELD_EXTRACTION_PROMPT,
    TICKET_FIELD_EXTRACTION_SYSTEM_PROMPT,
    TICKET_INTENT_CLASSIFICATION_PROMPT,
    TICKET_INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    TicketAgentIntentClassification,
    TicketAgentModelDependencies,
    TicketAgentModelOutputFailure,
    TicketAgentModelOutputFailureAction,
    TicketAgentModelOutputFailureKind,
    TicketAgentPromptName,
    TicketAgentRoute,
    TicketAgentPromptSpec,
    TicketAgentState,
    TicketAgentStreamPart,
    OrderQueryExecutor,
    TicketConfirmationRoute,
    TicketConfirmationStatus,
    TicketCreationStatus,
    TicketCreator,
    TicketFieldCompletionRoute,
    TicketFieldExtractionSource,
    TicketFieldExtractor,
    TicketFields,
    TicketIntent,
    TicketIntentClassifier,
    TicketIssueType,
    TicketNeedDecision,
    TicketNeedRoute,
    TicketNeedSource,
    TicketOrderQueryFailure,
    TicketOrderQueryFailureAction,
    TicketOrderQueryFailureKind,
    TicketOrderQueryStatus,
    TicketUrgencyLevel,
    TicketWriteSafetyStatus,
)

def get_ticket_agent_prompt_spec(
    prompt_name: TicketAgentPromptName,
) -> TicketAgentPromptSpec:
    return TICKET_AGENT_PROMPTS[prompt_name]


def _has_pydantic_error_type(details: object, error_type: str) -> bool:
    if not isinstance(details, list):
        return False

    return any(
        isinstance(error, dict) and error.get("type") == error_type
        for error in details
    )


def classify_ticket_agent_model_output_failure(
    exc: Exception,
) -> TicketAgentModelOutputFailure:
    if not isinstance(exc, AppException):
        return TicketAgentModelOutputFailure(
            code=type(exc).__name__,
            kind="unknown_error",
            action="raise_error",
            message="模型调用遇到未知异常",
            retryable=False,
        )

    if exc.code in TICKET_AGENT_MODEL_EMPTY_RESPONSE_CODES:
        return TicketAgentModelOutputFailure(
            code=exc.code,
            kind="empty_response",
            action="fallback_to_rule_based",
            message=exc.message,
            retryable=True,
        )

    if exc.code in TICKET_AGENT_MODEL_SCHEMA_VALIDATION_FAILED_CODES:
        kind: TicketAgentModelOutputFailureKind = (
            "invalid_json"
            if _has_pydantic_error_type(exc.details, "json_invalid")
            else "schema_validation"
        )
        return TicketAgentModelOutputFailure(
            code=exc.code,
            kind=kind,
            action="fallback_to_rule_based",
            message=exc.message,
            retryable=kind == "invalid_json",
        )

    if exc.code in TICKET_AGENT_MODEL_TRANSIENT_PROVIDER_ERROR_CODES:
        return TicketAgentModelOutputFailure(
            code=exc.code,
            kind="provider_error",
            action="fallback_to_rule_based",
            message=exc.message,
            retryable=True,
        )

    if exc.code in TICKET_AGENT_MODEL_CONFIGURATION_ERROR_CODES:
        return TicketAgentModelOutputFailure(
            code=exc.code,
            kind="configuration_error",
            action="raise_error",
            message=exc.message,
            retryable=False,
        )

    return TicketAgentModelOutputFailure(
        code=exc.code,
        kind="unknown_error",
        action="raise_error",
        message=exc.message,
        retryable=False,
    )


def get_ticket_intent_classification_json_schema() -> dict[str, Any]:
    return LLMTicketIntentClassification.model_json_schema()


def build_ticket_intent_classification_messages(
    user_message: str,
    *,
    prompt_spec: TicketAgentPromptSpec = TICKET_INTENT_CLASSIFICATION_PROMPT,
    history: list[str] | None = None,
) -> list[dict[str, str]]:
    schema_text = json.dumps(
        get_ticket_intent_classification_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    history_section = ""
    if history:
        history_section = "最近对话：\n" + "\n".join(history) + "\n\n"
    return [
        {
            "role": "system",
            "content": prompt_spec.system_prompt,
        },
        {
            "role": "user",
            "content": (
                "请把下面的用户消息分类成 JSON。\n"
                f"JSON Schema:\n{schema_text}\n\n"
                f"{history_section}用户消息:\n{user_message}"
            ),
        },
    ]


def parse_ticket_intent_classification_json(
    raw_json: str,
) -> TicketAgentIntentClassification:
    if not isinstance(raw_json, str) or not raw_json.strip():
        raise AppException(
            code="TICKET_INTENT_LLM_EMPTY_RESPONSE",
            message="模型没有返回可解析的意图识别结果",
            status_code=502,
        )

    try:
        result = LLMTicketIntentClassification.model_validate_json(raw_json)
    except ValidationError as exc:
        raise AppException(
            code="TICKET_INTENT_LLM_VALIDATION_FAILED",
            message="模型意图识别结果校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False),
        ) from exc

    return {
        "intent": result.intent,
        "reason": result.reason,
    }


class RuleBasedTicketIntentClassifier:
    def classify_intent(self, message: str, *, history: list[str] | None = None) -> TicketAgentIntentClassification:
        return classify_ticket_intent(message)


class FakeLLMTicketIntentClassifier:
    def classify_intent(self, message: str, *, history: list[str] | None = None) -> TicketAgentIntentClassification:
        classification = classify_ticket_intent(message)
        raw_json = json.dumps(classification, ensure_ascii=False)
        return parse_ticket_intent_classification_json(raw_json)


class LLMTicketIntentClassifier:
    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        prompt_spec: TicketAgentPromptSpec = TICKET_INTENT_CLASSIFICATION_PROMPT,
    ) -> None:
        self.settings = settings
        self._client = client
        self.prompt_spec = prompt_spec

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
        classification: TicketAgentIntentClassification,
    ) -> None:
        usage = extract_token_usage(completion)
        _record_ticket_agent_cost(self.settings, usage, classification["intent"])
        logger.info(
            (
                "ticket_intent_llm_classification_succeeded provider=%s model=%s "
                "prompt_name=%s prompt_version=%s elapsed_ms=%.2f intent=%s "
                "prompt_tokens=%s completion_tokens=%s total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            self.prompt_spec.name,
            self.prompt_spec.version,
            elapsed_ms,
            classification["intent"],
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
                "ticket_intent_llm_classification_failed code=%s provider=%s "
                "model=%s prompt_name=%s prompt_version=%s status_code=%s "
                "elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            self.prompt_spec.name,
            self.prompt_spec.version,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )

    def classify_intent(self, message: str, *, history: list[str] | None = None) -> TicketAgentIntentClassification:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_ticket_intent_classification_messages(
            message,
            prompt_spec=self.prompt_spec,
            history=history,
        )
        start_time = perf_counter()
        try:
            from app.agents.tracing_spans import set_span_attributes, start_llm_span

            with start_llm_span(
                model=self.settings.llm_model,
                provider=self.settings.llm_provider,
                prompt_name=self.prompt_spec.name,
            ):
                completion = self._get_client().chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                usage = extract_token_usage(completion)
                token_attributes: dict[str, int] = {}
                if usage.prompt_tokens is not None:
                    token_attributes["prompt_tokens"] = usage.prompt_tokens
                if usage.completion_tokens is not None:
                    token_attributes["completion_tokens"] = usage.completion_tokens
                if usage.total_tokens is not None:
                    token_attributes["total_tokens"] = usage.total_tokens
                if token_attributes:
                    set_span_attributes(token_attributes)
            raw_reply = extract_first_reply(completion)
            classification = parse_ticket_intent_classification_json(raw_reply)
        except AppException as exc:
            self._log_failure(exc, _elapsed_ms_since(start_time))
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(
                app_exception,
                _elapsed_ms_since(start_time),
                exc_info=True,
            )
            raise app_exception from exc

        self._log_success(_elapsed_ms_since(start_time), completion, classification)
        return classification


def create_llm_ticket_intent_classifier(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    prompt_spec: TicketAgentPromptSpec = TICKET_INTENT_CLASSIFICATION_PROMPT,
) -> LLMTicketIntentClassifier:
    return LLMTicketIntentClassifier(
        settings or get_settings(),
        client=client,
        prompt_spec=prompt_spec,
    )


def get_ticket_field_extraction_json_schema() -> dict[str, Any]:
    return LLMTicketFields.model_json_schema()


def build_ticket_field_extraction_messages(
    state: TicketAgentState,
    *,
    prompt_spec: TicketAgentPromptSpec = TICKET_FIELD_EXTRACTION_PROMPT,
) -> list[dict[str, str]]:
    schema_text = json.dumps(
        get_ticket_field_extraction_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    context = {
        "intent": state.get("intent"),
        "ticket_need_source": state.get("ticket_need_source"),
        "rag_answer_status": state.get("rag_answer_status"),
        "rag_no_context_reason": state.get("rag_no_context_reason"),
    }
    context_text = json.dumps(
        {key: value for key, value in context.items() if value is not None},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    normalized_message = state.get("normalized_message", "").strip()

    return [
        {
            "role": "system",
            "content": prompt_spec.system_prompt,
        },
        {
            "role": "user",
            "content": (
                "请把下面的 Agent 上下文和用户消息提取成工单字段 JSON。\n"
                f"JSON Schema:\n{schema_text}\n\n"
                f"Agent 上下文:\n{context_text}\n\n"
                f"用户消息:\n{normalized_message}"
            ),
        },
    ]


def parse_ticket_field_extraction_json(raw_json: str) -> TicketFields:
    if not isinstance(raw_json, str) or not raw_json.strip():
        raise AppException(
            code="TICKET_FIELD_LLM_EMPTY_RESPONSE",
            message="模型没有返回可解析的工单字段提取结果",
            status_code=502,
        )

    try:
        result = LLMTicketFields.model_validate_json(raw_json)
    except ValidationError as exc:
        raise AppException(
            code="TICKET_FIELD_LLM_VALIDATION_FAILED",
            message="模型工单字段提取结果校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False),
        ) from exc

    return {
        "issue_type": result.issue_type,
        "order_id": result.order_id,
        "description": result.description,
        "user_request": result.user_request,
        "urgency": result.urgency,
        "need_human_review": result.need_human_review,
    }


class RuleBasedTicketFieldExtractor:
    extraction_source: TicketFieldExtractionSource = "rule_based"

    def extract_fields(self, state: TicketAgentState) -> TicketFields:
        return extract_ticket_fields(state)


class FakeLLMTicketFieldExtractor:
    extraction_source: TicketFieldExtractionSource = "fake_llm"

    def extract_fields(self, state: TicketAgentState) -> TicketFields:
        fields = extract_ticket_fields(state)
        raw_json = json.dumps(fields, ensure_ascii=False)
        return parse_ticket_field_extraction_json(raw_json)


class LLMTicketFieldExtractor:
    extraction_source: TicketFieldExtractionSource = "llm"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        prompt_spec: TicketAgentPromptSpec = TICKET_FIELD_EXTRACTION_PROMPT,
    ) -> None:
        self.settings = settings
        self._client = client
        self.prompt_spec = prompt_spec

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
        fields: TicketFields,
    ) -> None:
        usage = extract_token_usage(completion)
        from app.services.cost_recorder import current_cost_intent

        _record_ticket_agent_cost(self.settings, usage, current_cost_intent())
        logger.info(
            (
                "ticket_field_llm_extraction_succeeded provider=%s model=%s "
                "prompt_name=%s prompt_version=%s elapsed_ms=%.2f issue_type=%s "
                "has_order_id=%s urgency=%s need_human_review=%s prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            self.prompt_spec.name,
            self.prompt_spec.version,
            elapsed_ms,
            fields["issue_type"],
            fields["order_id"] is not None,
            fields["urgency"],
            fields["need_human_review"],
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
                "ticket_field_llm_extraction_failed code=%s provider=%s "
                "model=%s prompt_name=%s prompt_version=%s status_code=%s "
                "elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            self.prompt_spec.name,
            self.prompt_spec.version,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )

    def extract_fields(self, state: TicketAgentState) -> TicketFields:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_ticket_field_extraction_messages(
            state,
            prompt_spec=self.prompt_spec,
        )
        start_time = perf_counter()
        try:
            from app.agents.tracing_spans import set_span_attributes, start_llm_span

            with start_llm_span(
                model=self.settings.llm_model,
                provider=self.settings.llm_provider,
                prompt_name=self.prompt_spec.name,
            ):
                completion = self._get_client().chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                usage = extract_token_usage(completion)
                token_attributes: dict[str, int] = {}
                if usage.prompt_tokens is not None:
                    token_attributes["prompt_tokens"] = usage.prompt_tokens
                if usage.completion_tokens is not None:
                    token_attributes["completion_tokens"] = usage.completion_tokens
                if usage.total_tokens is not None:
                    token_attributes["total_tokens"] = usage.total_tokens
                if token_attributes:
                    set_span_attributes(token_attributes)
            raw_reply = extract_first_reply(completion)
            fields = parse_ticket_field_extraction_json(raw_reply)
        except AppException as exc:
            self._log_failure(exc, _elapsed_ms_since(start_time))
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(
                app_exception,
                _elapsed_ms_since(start_time),
                exc_info=True,
            )
            raise app_exception from exc

        self._log_success(_elapsed_ms_since(start_time), completion, fields)
        return fields


def create_llm_ticket_field_extractor(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    prompt_spec: TicketAgentPromptSpec = TICKET_FIELD_EXTRACTION_PROMPT,
) -> LLMTicketFieldExtractor:
    return LLMTicketFieldExtractor(
        settings or get_settings(),
        client=client,
        prompt_spec=prompt_spec,
    )


def log_ticket_agent_model_output_fallback(
    *,
    component: str,
    failure: TicketAgentModelOutputFailure,
) -> None:
    logger.warning(
        (
            "ticket_agent_model_output_fallback component=%s code=%s kind=%s "
            "action=%s retryable=%s"
        ),
        component,
        failure.code,
        failure.kind,
        failure.action,
        failure.retryable,
    )


class ModelOutputFallbackTicketIntentClassifier:
    def __init__(
        self,
        primary: TicketIntentClassifier,
        *,
        fallback: TicketIntentClassifier | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or RuleBasedTicketIntentClassifier()

    def classify_intent(self, message: str, *, history: list[str] | None = None) -> TicketAgentIntentClassification:
        try:
            if history:
                return self.primary.classify_intent(message, history=history)
            return self.primary.classify_intent(message)
        except Exception as exc:
            failure = classify_ticket_agent_model_output_failure(exc)
            if failure.action != "fallback_to_rule_based":
                raise

            log_ticket_agent_model_output_fallback(
                component="intent_classifier",
                failure=failure,
            )
            if history:
                return self.fallback.classify_intent(message, history=history)
            return self.fallback.classify_intent(message)


class ModelOutputFallbackTicketFieldExtractor:
    extraction_source: TicketFieldExtractionSource = "llm"

    def __init__(
        self,
        primary: TicketFieldExtractor,
        *,
        fallback: TicketFieldExtractor | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or RuleBasedTicketFieldExtractor()
        self.last_extraction_source: TicketFieldExtractionSource = (
            primary.extraction_source
        )

    def extract_fields(self, state: TicketAgentState) -> TicketFields:
        try:
            fields = self.primary.extract_fields(state)
            self.last_extraction_source = self.primary.extraction_source
            return fields
        except Exception as exc:
            failure = classify_ticket_agent_model_output_failure(exc)
            if failure.action != "fallback_to_rule_based":
                raise

            log_ticket_agent_model_output_fallback(
                component="field_extractor",
                failure=failure,
            )
            self.last_extraction_source = "llm_fallback_rule_based"
            return self.fallback.extract_fields(state)


def get_ticket_field_extraction_source(
    extractor: TicketFieldExtractor,
) -> TicketFieldExtractionSource:
    return getattr(extractor, "last_extraction_source", extractor.extraction_source)


def ensure_real_ticket_agent_llm_is_configured(settings: Settings) -> None:
    if not settings.has_llm_api_key:
        raise AppException(
            code="LLM_API_KEY_MISSING",
            message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
            status_code=500,
        )


def create_ticket_agent_model_dependencies(
    mode: TicketAgentModelMode | None = None,
    *,
    settings: Settings | None = None,
    client: Any | None = None,
    intent_prompt_spec: TicketAgentPromptSpec = TICKET_INTENT_CLASSIFICATION_PROMPT,
    field_prompt_spec: TicketAgentPromptSpec = TICKET_FIELD_EXTRACTION_PROMPT,
    enable_model_output_fallback: bool = False,
) -> TicketAgentModelDependencies:
    selected_settings = settings or get_settings()
    selected_mode = mode or selected_settings.ticket_agent_model_mode

    if selected_mode == "rule_based":
        return {
            "mode": "rule_based",
            "intent_classifier": None,
            "field_extractor": None,
            "emotion_classifier": None,
        }

    if selected_mode == "fake_llm":
        return {
            "mode": "fake_llm",
            "intent_classifier": FakeLLMTicketIntentClassifier(),
            "field_extractor": FakeLLMTicketFieldExtractor(),
            "emotion_classifier": FakeLLMEmotionClassifier(),
        }

    if selected_mode == "real_llm":
        ensure_real_ticket_agent_llm_is_configured(selected_settings)
        intent_classifier: TicketIntentClassifier = create_llm_ticket_intent_classifier(
            selected_settings,
            client=client,
            prompt_spec=intent_prompt_spec,
        )
        field_extractor: TicketFieldExtractor = create_llm_ticket_field_extractor(
            selected_settings,
            client=client,
            prompt_spec=field_prompt_spec,
        )
        if enable_model_output_fallback:
            intent_classifier = ModelOutputFallbackTicketIntentClassifier(
                intent_classifier,
            )
            field_extractor = ModelOutputFallbackTicketFieldExtractor(
                field_extractor,
            )

        return {
            "mode": "real_llm",
            "intent_classifier": intent_classifier,
            "field_extractor": field_extractor,
            "emotion_classifier": LLMEmotionClassifier(selected_settings, client=client),
        }

    raise ValueError(f"Unsupported ticket agent model mode: {selected_mode}")


class FakePolicyRagService:
    def answer_policy_question(self, query: str, *, access_scope=None) -> RagAnswer:
        normalized_query = query.strip()
        lowered_query = normalized_query.casefold()

        if not normalized_query:
            return build_no_context_rag_answer()

        if "退款" in lowered_query:
            return build_grounded_rag_answer(
                "根据知识库，退款申请通常需要先核对订单状态和售后条件，"
                "用户可以按退款退货规则提交申请。",
                [
                    _make_fake_retrieved_chunk(
                        chunk_id="refund_return_policy_chunk_0001",
                        content="退款申请通常需要先核对订单状态、售后条件和商品状态。",
                        source="refund-return-policy.md",
                        title="退款退货规则",
                        section="退款申请",
                    )
                ],
            )

        if "退货" in lowered_query:
            return build_grounded_rag_answer(
                "根据知识库，退货通常需要商品符合售后规则，并按页面或客服指引提交退货申请。",
                [
                    _make_fake_retrieved_chunk(
                        chunk_id="refund_return_policy_chunk_0002",
                        content="退货通常需要商品符合售后规则，并按指引提交退货申请。",
                        source="refund-return-policy.md",
                        title="退款退货规则",
                        section="退货申请",
                    )
                ],
            )

        if (
            "账号安全" in lowered_query
            or "异常登录" in lowered_query
            or "身份验证" in lowered_query
        ):
            return build_grounded_rag_answer(
                "根据知识库，账号安全相关操作通常需要进行身份验证，"
                "客服不能在聊天中索要完整敏感身份信息。",
                [
                    _make_fake_retrieved_chunk(
                        chunk_id="account_security_faq_chunk_0001",
                        content="账号安全相关操作通常需要身份验证，客服不能索要完整敏感身份信息。",
                        source="account-security-faq.md",
                        title="账号安全常见问题",
                        section="身份验证",
                    )
                ],
            )

        return build_no_context_rag_answer()


def normalize_user_input_node(state: TicketAgentState) -> TicketAgentState:
    user_message = state.get("user_message", "")

    return {
        "normalized_message": user_message.strip(),
        # 情绪每轮独立识别：重置上一轮字段，避免带 checkpointer 时跨轮残留
        "emotion": None,
        "emotion_reason": None,
        "emotion_handoff_requested": False,
        "node_history": ["normalize_user_input"],
    }


def _is_cancel_request(lowered_message: str, normalized_message: str) -> bool:
    """Detect an explicit cancel-order action versus a passive policy question.

    Cancel detection runs before refund detection: 取消订单/退单/取消购物 express
    a cancel request, not a refund.  Action phrases take priority over query
    words when an order id is present ("取消订单 A1002，可以吗？"), while bare
    policy consultations ("取消政策是什么"/"怎么取消订单"/"取消多久生效") fall
    through to policy_question.
    """
    has_order_id = _extract_order_id(normalized_message) is not None
    querying = _contains_any(lowered_message, CANCEL_QUERY_WORDS)
    policy_hint = _contains_any(lowered_message, POLICY_KEYWORDS)

    if _contains_any(lowered_message, CANCEL_ACTION_PHRASES):
        return not querying or has_order_id
    if (
        has_order_id
        and not querying
        and not policy_hint
        and _contains_any(lowered_message, CANCEL_KEYWORDS)
    ):
        return True
    if CANCEL_WITH_ORDER_PATTERN.search(lowered_message) is not None:
        return True
    return False


def _is_refund_request(lowered_message: str, normalized_message: str) -> bool:
    """Detect an explicit refund action versus a passive policy question.

    Action phrases like 申请退款/直接退款 are clear refund requests — and take
    priority over query words, so "帮我退款，订单 A1002 可以吗？" still routes to
    refund_request (explicit action + concrete order id).  Query words only
    demote a bare policy consultation ("怎么申请退款") that carries no order id.
    A refund keyword combined with a concrete order id (订单 A1002 我要退款) or a
    bare 退 + order id (退 A1002 的款 / 退A1002) also implies an action.  Pure
    policy questions (退款政策是什么/退款规则/查订单 A1001 的退款政策) fall
    through to policy_question.
    """
    has_order_id = _extract_order_id(normalized_message) is not None
    querying = _contains_any(lowered_message, REFUND_QUERY_WORDS)
    policy_hint = _contains_any(lowered_message, POLICY_KEYWORDS)

    if _contains_any(lowered_message, REFUND_ACTION_PHRASES):
        return not querying or has_order_id
    if (
        has_order_id
        and not querying
        and not policy_hint
        and _contains_any(lowered_message, REFUND_KEYWORDS)
    ):
        return True
    if REFUND_WITH_ORDER_PATTERN.search(lowered_message) is not None:
        return True
    return False


def classify_ticket_intent(message: str) -> TicketAgentIntentClassification:
    normalized_message = message.strip()
    lowered_message = normalized_message.casefold()

    if not normalized_message:
        return {
            "intent": "unclear",
            "reason": "用户输入为空，需要先追问用户要处理的问题。",
        }

    if _contains_any(lowered_message, UNSUPPORTED_KEYWORDS):
        return {
            "intent": "unsupported",
            "reason": "用户请求超出当前客服 Agent v1 的安全业务范围。",
        }

    if _is_cancel_request(lowered_message, normalized_message):
        return {
            "intent": "cancel_request",
            "reason": "用户明确要求取消订单并可能提供了订单号，需要进入取消确认流程。",
        }

    if _is_refund_request(lowered_message, normalized_message):
        return {
            "intent": "refund_request",
            "reason": "用户明确要求执行退款并可能提供了订单号，需要进入退款确认流程。",
        }

    if _contains_any(lowered_message, SMALLTALK_KEYWORDS):
        return {
            "intent": "smalltalk",
            "reason": "用户在进行问候或询问助手能力，不需要查询业务系统。",
        }

    if _contains_any(lowered_message, TICKET_KEYWORDS):
        return {
            "intent": "ticket_request",
            "reason": "用户表达了投诉、售后处理或创建工单诉求。",
        }

    if (
        _contains_any(lowered_message, ORDER_KEYWORDS)
        and not _contains_any(lowered_message, POLICY_KEYWORDS)
    ):
        return {
            "intent": "order_query",
            "reason": "用户在询问订单、物流、支付或发货状态。",
        }

    if _contains_any(lowered_message, POLICY_KEYWORDS):
        return {
            "intent": "policy_question",
            "reason": "用户在询问规则、政策或 FAQ 类知识库问题。",
        }

    if normalized_message in UNCLEAR_MESSAGES:
        return {
            "intent": "unclear",
            "reason": "用户描述过于笼统，需要追问具体问题和必要信息。",
        }

    return {
        "intent": "unclear",
        "reason": "当前规则分类器无法稳定判断意图，需要追问用户补充信息。",
    }


def emotion_recognition_node(
    state: TicketAgentState,
    classifier: CustomerEmotionClassifier,
) -> TicketAgentState:
    message = state.get("normalized_message") or state.get("user_message") or ""
    result = classifier.recognize(message)
    return {
        "emotion": result.emotion,
        "emotion_reason": result.reason,
        "emotion_handoff_requested": result.emotion in EMOTION_HANDOFF_EMOTIONS,
        "node_history": ["emotion_recognition"],
    }


def route_by_emotion(state: TicketAgentState) -> str:
    if state.get("emotion_handoff_requested"):
        return "handoff"
    return "continue"


EMOTION_ROUTES = {
    "handoff": "build_emotion_handoff_answer",
    "continue": "classify_intent",
}


def build_emotion_handoff_answer_node(state: TicketAgentState) -> TicketAgentState:
    emotion = state.get("emotion", CustomerEmotion.NEUTRAL)
    reason = state.get("emotion_reason") or ""
    emotion_label = {
        CustomerEmotion.ANGRY: "愤怒",
        CustomerEmotion.ANXIOUS: "焦虑",
        CustomerEmotion.URGENT: "急切",
    }.get(emotion, str(emotion.value))
    reply = (
        f"非常抱歉给您带来不好的体验。我注意到您现在的情绪比较{emotion_label}（{reason}），"
        "已为您优先转接人工客服，请稍候，人工客服会尽快为您处理。"
    )
    return {
        "final_answer": reply,
        "emotion_handoff_requested": True,
        "node_history": ["build_emotion_handoff_answer"],
    }


def _record_ticket_agent_cost(settings: Any, usage: Any, intent: str) -> None:
    """Record an agent LLM call into the cost aggregator (best effort)."""
    try:
        from app.core.token_usage import TokenPricing, estimate_token_cost
        from app.services.cost_recorder import get_cost_recorder

        pricing: TokenPricing | None = None
        if settings.has_llm_token_pricing:
            pricing = TokenPricing(
                input_cost_per_million_tokens=settings.llm_input_cost_per_million_tokens or 0.0,
                output_cost_per_million_tokens=settings.llm_output_cost_per_million_tokens or 0.0,
            )
        # Token 用量始终记录；费用在 pricing 缺失时为 0（missing_pricing）
        estimate = estimate_token_cost(usage, pricing)
        get_cost_recorder().record(
            model=settings.llm_model,
            intent=intent,
            input_tokens=getattr(usage, "prompt_tokens", None) or 0,
            output_tokens=getattr(usage, "completion_tokens", None) or 0,
            estimated_cost=float(estimate.total_cost or 0.0),
        )
    except Exception:  # pragma: no cover - cost tracking must never break chat
        logger.debug("ticket_agent_cost_record_failed", exc_info=True)


def classify_intent_node(
    state: TicketAgentState,
    *,
    classifier: TicketIntentClassifier | None = None,
) -> TicketAgentState:
    normalized_message = state.get("normalized_message", "")
    if has_active_cancel_collection(state):
        return {
            "intent": "cancel_request",
            "intent_reason": "用户正在补充上一轮取消订单流程缺少的信息。",
            "node_history": ["classify_intent"],
        }
    if has_active_refund_collection(state):
        return {
            "intent": "refund_request",
            "intent_reason": "用户正在补充上一轮退款流程缺少的信息。",
            "node_history": ["classify_intent"],
        }
    if has_active_ticket_field_collection(state):
        return {
            "intent": "ticket_request",
            "intent_reason": "用户正在补充上一轮工单流程缺少的信息。",
            "node_history": ["classify_intent"],
        }

    history = state.get("history")
    if classifier is not None:
        if history:
            classification = classifier.classify_intent(normalized_message, history=history)
        else:
            classification = classifier.classify_intent(normalized_message)
    else:
        classification = classify_ticket_intent(normalized_message)

    return {
        "intent": classification["intent"],
        "intent_reason": classification["reason"],
        "node_history": ["classify_intent"],
    }


def route_by_intent(state: TicketAgentState) -> TicketAgentRoute:
    intent = state.get("intent")
    if intent:
        from app.services.cost_recorder import set_cost_intent

        set_cost_intent(intent)
    if intent in TICKET_AGENT_INTENT_ROUTES:
        return intent
    return "unclear"


def decide_ticket_need(state: TicketAgentState) -> TicketNeedDecision:
    intent = state.get("intent")
    rag_answer_status = state.get("rag_answer_status")

    if intent == "ticket_request":
        return {
            "needs_ticket": True,
            "reason": "用户明确表达了投诉、售后处理或创建工单诉求，需要进入工单流程。",
            "source": "explicit_user_request",
        }

    if intent == "policy_question" and rag_answer_status == "no_context":
        return {
            "needs_ticket": True,
            "reason": "知识库没有找到足够资料，需要进入工单流程记录问题或交给人工处理。",
            "source": "rag_no_context",
        }

    if intent == "policy_question" and rag_answer_status == "answered":
        return {
            "needs_ticket": False,
            "reason": "知识库已给出可引用回答，当前不需要创建工单。",
            "source": "rag_answered",
        }

    return {
        "needs_ticket": False,
        "reason": "当前路线暂不需要创建工单。",
        "source": "not_applicable",
    }


def decide_ticket_need_node(state: TicketAgentState) -> TicketAgentState:
    decision = decide_ticket_need(state)

    return {
        "needs_ticket": decision["needs_ticket"],
        "ticket_need_reason": decision["reason"],
        "ticket_need_source": decision["source"],
        "node_history": ["decide_ticket_need"],
    }


def route_by_ticket_need(state: TicketAgentState) -> TicketNeedRoute:
    if state.get("needs_ticket") is True:
        return "create_ticket"
    return "finish"


def extract_ticket_fields(state: TicketAgentState) -> TicketFields:
    normalized_message = state.get("normalized_message", "").strip()
    lowered_message = normalized_message.casefold()
    ticket_need_source = state.get("ticket_need_source")
    rag_answer_status = state.get("rag_answer_status")
    issue_type = _infer_ticket_issue_type(
        lowered_message,
        ticket_need_source=ticket_need_source,
        rag_answer_status=rag_answer_status,
    )
    urgency = _infer_ticket_urgency(lowered_message, issue_type=issue_type)

    return {
        "issue_type": issue_type,
        "order_id": _extract_order_id(normalized_message),
        "description": _build_ticket_description(
            normalized_message,
            ticket_need_source=ticket_need_source,
        ),
        "user_request": _infer_ticket_user_request(
            lowered_message,
            issue_type=issue_type,
            ticket_need_source=ticket_need_source,
        ),
        "urgency": urgency,
        "need_human_review": (
            ticket_need_source in {"explicit_user_request", "rag_no_context"}
            or urgency == "high"
        ),
    }


def has_active_ticket_field_collection(state: TicketAgentState) -> bool:
    return (
        state.get("needs_ticket") is True
        and isinstance(state.get("ticket_fields"), dict)
        and bool(state.get("missing_ticket_fields"))
    )


def has_active_refund_collection(state: TicketAgentState) -> bool:
    return (
        state.get("refund_request_active") is True
        and isinstance(state.get("ticket_fields"), dict)
        and bool(state.get("missing_ticket_fields"))
    )


def has_active_cancel_collection(state: TicketAgentState) -> bool:
    return (
        state.get("cancel_request_active") is True
        and isinstance(state.get("ticket_fields"), dict)
        and bool(state.get("missing_ticket_fields"))
    )


def merge_ticket_fields(
    previous_fields: TicketFields | None,
    latest_fields: TicketFields,
) -> TicketFields:
    if previous_fields is None:
        return latest_fields

    return {
        "issue_type": (
            latest_fields["issue_type"]
            if latest_fields["issue_type"] != "unknown"
            else previous_fields["issue_type"]
        ),
        "order_id": latest_fields["order_id"] or previous_fields["order_id"],
        "description": previous_fields["description"] or latest_fields["description"],
        "user_request": previous_fields["user_request"] or latest_fields["user_request"],
        "urgency": (
            "high"
            if latest_fields["urgency"] == "high"
            else previous_fields["urgency"]
        ),
        "need_human_review": (
            previous_fields["need_human_review"]
            or latest_fields["need_human_review"]
        ),
    }


def find_missing_ticket_fields(fields: TicketFields) -> list[str]:
    missing_fields: list[str] = []

    if fields["issue_type"] == "unknown":
        missing_fields.append("issue_type")
    if not fields["description"].strip():
        missing_fields.append("description")
    if not fields["user_request"].strip():
        missing_fields.append("user_request")
    if (
        fields["issue_type"] in ORDER_REQUIRED_ISSUE_TYPES
        and fields["order_id"] is None
    ):
        missing_fields.append("order_id")

    return missing_fields


def route_by_ticket_fields_complete(state: TicketAgentState) -> TicketFieldCompletionRoute:
    if state.get("ticket_fields_complete") is True:
        return "request_confirmation"
    return "ask_missing_fields"


def route_by_ticket_confirmation(state: TicketAgentState) -> TicketConfirmationRoute:
    if state.get("ticket_confirmation_correction_requested") is True:
        return "request_confirmation"
    if state.get("ticket_confirmation_approved") is True:
        if state.get("cancel_request_active") is True:
            return "execute_cancel_request"
        if state.get("refund_request_active") is True:
            return "execute_refund_request"
        return "execute_create_ticket"
    return "finish"


def build_missing_ticket_fields_question(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "工单字段已经完整，后续课程会学习如何请求用户确认。"

    field_questions = [
        MISSING_TICKET_FIELD_QUESTIONS.get(field, f"请补充 {field}。")
        for field in missing_fields
    ]

    if len(field_questions) == 1:
        return field_questions[0]

    return "为了继续创建工单，请补充以下信息：" + "；".join(field_questions)


def build_ticket_confirmation_id(fields: TicketFields) -> str:
    confirmation_payload = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(confirmation_payload.encode("utf-8")).hexdigest()[:32]


def build_ticket_confirmation_message(fields: TicketFields) -> str:
    issue_type_label = TICKET_ISSUE_TYPE_LABELS[fields["issue_type"]]
    urgency_label = TICKET_URGENCY_LABELS[fields["urgency"]]
    order_id = fields["order_id"] or "无"
    human_review = "是" if fields["need_human_review"] else "否"

    return (
        "我已整理好一份待确认工单，请确认是否按以下信息创建：\n"
        f"问题类型：{issue_type_label}\n"
        f"订单号：{order_id}\n"
        f"问题描述：{fields['description']}\n"
        f"用户诉求：{fields['user_request']}\n"
        f"紧急程度：{urgency_label}\n"
        f"是否需要人工复核：{human_review}\n"
        "如果信息正确，请回复“确认创建”；如果不正确，请说明需要修改的内容。"
    )


def build_pending_ticket_confirmation(fields: TicketFields) -> PendingTicketConfirmation:
    issue_type_label = TICKET_ISSUE_TYPE_LABELS[fields["issue_type"]]
    order_id = fields["order_id"] or "无订单号"
    summary = f"{issue_type_label}，{order_id}，{fields['user_request']}"

    return {
        "confirmation_id": build_ticket_confirmation_id(fields),
        "status": "pending",
        "title": f"待确认工单：{issue_type_label}",
        "summary": summary,
        "ticket_fields": fields,
        "message": build_ticket_confirmation_message(fields),
    }


def build_ticket_confirmation_interrupt_payload(
    pending_confirmation: PendingTicketConfirmation,
    *,
    is_refund_execution: bool = False,
    is_cancel_execution: bool = False,
) -> dict[str, Any]:
    return {
        "kind": TICKET_CONFIRMATION_INTERRUPT_KIND,
        "confirmation_id": pending_confirmation["confirmation_id"],
        "message": pending_confirmation["message"],
        "is_refund_execution": is_refund_execution,
        "is_cancel_execution": is_cancel_execution,
        "pending_ticket_confirmation": pending_confirmation,
    }


def is_ticket_confirmation_resume_approved(resume_value: Any) -> bool:
    if isinstance(resume_value, bool):
        return resume_value
    if isinstance(resume_value, dict):
        return resume_value.get("approved") is True
    return False


def get_ticket_confirmation_resume_actor_id(resume_value: Any) -> str | None:
    if not isinstance(resume_value, dict):
        return None

    actor_id = resume_value.get("actor_id")
    if not isinstance(actor_id, str):
        return None

    normalized_actor_id = actor_id.strip()
    return normalized_actor_id or None


def get_ticket_confirmation_resume_corrected_fields(
    resume_value: Any,
) -> TicketFields | None:
    if not isinstance(resume_value, dict):
        return None

    fields = resume_value.get("corrected_ticket_fields")
    if not isinstance(fields, dict):
        return None

    return fields


def build_create_ticket_args_from_fields(
    fields: TicketFields,
    *,
    actor_id: str,
) -> CreateTicketArgs:
    category = TICKET_ISSUE_TYPE_TO_CATEGORY.get(fields["issue_type"])
    if category is None:
        raise AppException(
            code="TICKET_FIELDS_INCOMPLETE",
            message="工单字段还不完整，暂时不能创建工单。",
            status_code=422,
        )

    return CreateTicketArgs(
        requester_id=actor_id,
        title=_build_ticket_creation_title(fields),
        description=fields["description"],
        category=category,
        priority=TICKET_URGENCY_TO_PRIORITY[fields["urgency"]],
        related_order_id=fields["order_id"],
    )


def build_ticket_agent_fallback_state(
    *,
    node_name: str,
    code: str = TICKET_AGENT_FALLBACK_ERROR_CODE,
    message: str = TICKET_AGENT_FALLBACK_MESSAGE,
) -> TicketAgentState:
    return {
        "agent_error_code": code,
        "agent_error_message": message,
        "agent_error_node": node_name,
        "fallback_used": True,
        "final_answer": message,
        "node_history": [node_name],
    }


def build_ticket_creation_failure_state(
    *,
    code: str,
    message: str,
) -> TicketAgentState:
    update = build_ticket_agent_fallback_state(
        node_name="create_ticket",
        code=code,
        message=message,
    )
    update.update(
        {
            "ticket_creation_status": "failed",
            "ticket_creation_error_code": code,
            "ticket_creation_error_message": message,
        }
    )
    return update


def build_ticket_write_safety_state(
    *,
    status: TicketWriteSafetyStatus,
    definition: ToolDefinition | None = None,
    idempotency_key: str | None = None,
) -> TicketAgentState:
    tool_definition = definition or get_tool_definition(CREATE_TICKET_TOOL_NAME)
    return {
        "ticket_tool_name": CREATE_TICKET_TOOL_NAME,
        "ticket_tool_access_level": (
            tool_definition.access_level.value if tool_definition is not None else None
        ),
        "ticket_tool_requires_confirmation": (
            tool_definition.requires_confirmation if tool_definition is not None else None
        ),
        "ticket_write_safety_status": status,
        "ticket_creation_idempotency_key": idempotency_key,
    }


def execute_ticket_order_query(arguments: QueryOrderArgs) -> QueryOrderResult:
    return run_query_order_tool(arguments)


def build_order_query_argument_validation_failure() -> TicketOrderQueryFailure:
    return TicketOrderQueryFailure(
        code=TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_ERROR_CODE,
        kind="argument_validation",
        action="ask_user_to_check_order_id",
        message=TICKET_ORDER_QUERY_ARGUMENT_VALIDATION_MESSAGE,
        retryable=False,
        status_code=422,
    )


def classify_ticket_order_query_failure(exc: Exception) -> TicketOrderQueryFailure:
    if not isinstance(exc, AppException):
        return TicketOrderQueryFailure(
            code=TICKET_ORDER_QUERY_UNEXPECTED_ERROR_CODE,
            kind="unknown_error",
            action="retry_later",
            message=TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE,
            retryable=True,
            status_code=502,
        )

    if exc.code in TICKET_ORDER_QUERY_NOT_FOUND_CODES:
        return TicketOrderQueryFailure(
            code=exc.code,
            kind="not_found",
            action="ask_user_to_check_order_id",
            message=exc.message,
            retryable=False,
            status_code=exc.status_code,
        )

    if exc.code in TICKET_ORDER_QUERY_TIMEOUT_CODES:
        return TicketOrderQueryFailure(
            code=exc.code,
            kind="timeout",
            action="retry_later",
            message=exc.message,
            retryable=True,
            status_code=exc.status_code,
        )

    if exc.code in TICKET_ORDER_QUERY_UPSTREAM_ERROR_CODES:
        return TicketOrderQueryFailure(
            code=exc.code,
            kind="upstream_error",
            action="retry_later",
            message=exc.message,
            retryable=True,
            status_code=exc.status_code,
        )

    if exc.code in TICKET_ORDER_QUERY_RESULT_VALIDATION_FAILED_CODES:
        return TicketOrderQueryFailure(
            code=exc.code,
            kind="result_validation",
            action="investigate_system",
            message=TICKET_ORDER_QUERY_RESULT_VALIDATION_MESSAGE,
            retryable=False,
            status_code=exc.status_code,
        )

    if exc.code in TICKET_ORDER_QUERY_TOOL_ERROR_CODES:
        return TicketOrderQueryFailure(
            code=exc.code,
            kind="tool_error",
            action="retry_later",
            message=TICKET_ORDER_QUERY_UNEXPECTED_ERROR_MESSAGE,
            retryable=True,
            status_code=exc.status_code,
        )

    return TicketOrderQueryFailure(
        code=exc.code,
        kind="tool_error",
        action="contact_human_support",
        message=exc.message,
        retryable=False,
        status_code=exc.status_code,
    )


def build_order_query_missing_order_id_state() -> TicketAgentState:
    return {
        "order_query_order_id": None,
        "order_query_status": "missing_order_id",
        "order_query_error_code": "ORDER_ID_REQUIRED",
        "order_query_error_kind": "missing_order_id",
        "order_query_error_action": "ask_user_for_order_id",
        "order_query_error_message": TICKET_ORDER_QUERY_MISSING_ORDER_ID_MESSAGE,
        "order_query_retryable": False,
        "order_query_error_status_code": None,
        "final_answer": TICKET_ORDER_QUERY_MISSING_ORDER_ID_MESSAGE,
        "node_history": ["query_order"],
    }


def build_order_query_failure_state(
    *,
    order_id: str,
    failure: TicketOrderQueryFailure,
) -> TicketAgentState:
    update = build_ticket_agent_fallback_state(
        node_name="query_order",
        code=failure.code,
        message=failure.message,
    )
    update.update(
        {
            "order_query_order_id": order_id,
            "order_query_status": "failed",
            "order_query_error_code": failure.code,
            "order_query_error_kind": failure.kind,
            "order_query_error_action": failure.action,
            "order_query_error_message": failure.message,
            "order_query_retryable": failure.retryable,
            "order_query_error_status_code": failure.status_code,
        }
    )
    return update


def build_order_query_success_answer(result: QueryOrderResult) -> str:
    order_status = ORDER_STATUS_LABELS.get(
        str(result.order_status),
        str(result.order_status),
    )
    payment_status = PAYMENT_STATUS_LABELS.get(
        str(result.payment_status),
        str(result.payment_status),
    )
    ticket_hint = (
        "如仍有售后问题，可以继续帮你整理工单。"
        if result.can_create_ticket
        else "当前订单暂不建议直接创建工单，可以先根据订单状态继续观察。"
    )
    return (
        f"查询到订单 {result.order_id}：\n"
        f"- 订单状态：{order_status}\n"
        f"- 支付状态：{payment_status}\n"
        f"- 物流摘要：{result.logistics_message}\n"
        f"- 最新事件：{result.latest_event}\n"
        f"- 数据来源：{result.source}\n"
        f"{ticket_hint}"
    )


def build_ticket_agent_observation_metadata(
    state: dict[str, Any],
    *,
    operation: str,
    thread_id: str | None = None,
    elapsed_ms: float | None = None,
) -> dict[str, Any]:
    node_history = list(state.get("node_history", []))
    metadata: dict[str, Any] = {
        "operation": operation,
        "trace_id": state.get("agent_trace_id") or get_trace_id(),
        "thread_id": _safe_log_value(thread_id),
        "intent": _safe_log_value(state.get("intent")),
        "node_count": len(node_history),
        "last_node": _safe_log_value(node_history[-1] if node_history else None),
        "interrupted": bool(state.get("__interrupt__")),
        "fallback_used": state.get("fallback_used") is True,
        "agent_error_code": _safe_log_value(state.get("agent_error_code")),
        "ticket_creation_status": _safe_log_value(
            state.get("ticket_creation_status")
        ),
    }
    if elapsed_ms is not None:
        metadata["elapsed_ms"] = round(elapsed_ms, 2)
    return metadata


def log_ticket_agent_run_started(
    *,
    operation: str,
    user_message: str | None = None,
    thread_id: str | None = None,
    actor_id: str | None = None,
) -> None:
    logger.info(
        (
            "ticket_agent_started operation=%s thread_id=%s actor_id=%s "
            "message_length=%s"
        ),
        operation,
        _safe_log_value(thread_id),
        _safe_log_value(actor_id),
        len(user_message or ""),
    )


def log_ticket_agent_run_finished(
    state: dict[str, Any],
    *,
    operation: str,
    elapsed_ms: float,
    thread_id: str | None = None,
) -> None:
    metadata = build_ticket_agent_observation_metadata(
        state,
        operation=operation,
        thread_id=thread_id,
        elapsed_ms=elapsed_ms,
    )
    logger.info(
        (
            "ticket_agent_finished operation=%s thread_id=%s elapsed_ms=%.2f "
            "intent=%s node_count=%s last_node=%s interrupted=%s "
            "fallback_used=%s agent_error_code=%s ticket_creation_status=%s"
        ),
        metadata["operation"],
        metadata["thread_id"],
        metadata["elapsed_ms"],
        metadata["intent"],
        metadata["node_count"],
        metadata["last_node"],
        metadata["interrupted"],
        metadata["fallback_used"],
        metadata["agent_error_code"],
        metadata["ticket_creation_status"],
    )


def log_ticket_agent_run_failed(
    exc: Exception,
    *,
    operation: str,
    elapsed_ms: float,
    thread_id: str | None = None,
) -> None:
    error_code = exc.code if isinstance(exc, AppException) else TICKET_AGENT_FALLBACK_ERROR_CODE
    logger.warning(
        (
            "ticket_agent_failed operation=%s thread_id=%s elapsed_ms=%.2f "
            "code=%s error_type=%s"
        ),
        operation,
        _safe_log_value(thread_id),
        elapsed_ms,
        error_code,
        type(exc).__name__,
    )


def _access_scope_from_state(state: TicketAgentState):
    if not state.get("actor_roles") and not state.get("actor_tenant_id"):
        return None
    from app.services.console_agent_service import build_rag_access_scope

    return build_rag_access_scope(
        user_id=state.get("ticket_actor_id", ""),
        tenant_id=state.get("actor_tenant_id") or "",
        roles=state.get("actor_roles") or (),
    )


def retrieve_policy_node(
    state: TicketAgentState,
    service: PolicyRagService | None = None,
) -> TicketAgentState:
    rag_query = state.get("normalized_message", "").strip()
    rag_service = service or create_policy_rag_service()
    access_scope = _access_scope_from_state(state)
    rag_answer = rag_service.answer_policy_question(rag_query, access_scope=access_scope)

    return {
        "rag_query": rag_query,
        "rag_answer_status": rag_answer.status.value,
        "rag_citations": [citation.model_dump() for citation in rag_answer.citations],
        "rag_no_context_reason": (
            rag_answer.no_context_reason.value
            if rag_answer.no_context_reason is not None
            else None
        ),
        "rag_suggestions": list(rag_answer.suggestions),
        "final_answer": rag_answer.answer,
        "node_history": ["retrieve_policy"],
    }


def query_order_node(
    state: TicketAgentState,
    *,
    order_query_executor: OrderQueryExecutor | None = None,
) -> TicketAgentState:
    normalized_message = state.get("normalized_message") or state.get("user_message", "")
    order_id = _extract_order_id(normalized_message)
    if order_id is None:
        logger.info(
            "ticket_agent_query_order_missing_order_id message_length=%s",
            len(normalized_message),
        )
        return build_order_query_missing_order_id_state()

    try:
        arguments = QueryOrderArgs(order_id=order_id)
    except ValidationError:
        failure = build_order_query_argument_validation_failure()
        logger.warning(
            (
                "ticket_agent_query_order_failed order_id=%s code=%s kind=%s "
                "action=%s retryable=%s"
            ),
            order_id,
            failure.code,
            failure.kind,
            failure.action,
            failure.retryable,
        )
        return build_order_query_failure_state(
            order_id=order_id,
            failure=failure,
        )

    executor = order_query_executor or execute_ticket_order_query
    logger.info("ticket_agent_query_order_started order_id=%s", arguments.order_id)
    try:
        result = executor(arguments)
    except AppException as exc:
        failure = classify_ticket_order_query_failure(exc)
        logger.warning(
            (
                "ticket_agent_query_order_failed order_id=%s code=%s kind=%s "
                "action=%s retryable=%s status_code=%s"
            ),
            arguments.order_id,
            failure.code,
            failure.kind,
            failure.action,
            failure.retryable,
            failure.status_code,
        )
        return build_order_query_failure_state(
            order_id=arguments.order_id,
            failure=failure,
        )
    except Exception as exc:
        failure = classify_ticket_order_query_failure(exc)
        logger.warning(
            (
                "ticket_agent_query_order_failed order_id=%s code=%s kind=%s "
                "action=%s retryable=%s error_type=%s"
            ),
            arguments.order_id,
            failure.code,
            failure.kind,
            failure.action,
            failure.retryable,
            type(exc).__name__,
            exc_info=True,
        )
        return build_order_query_failure_state(
            order_id=arguments.order_id,
            failure=failure,
        )

    logger.info(
        "ticket_agent_query_order_succeeded order_id=%s source=%s",
        result.order_id,
        result.source,
    )
    return {
        "order_query_order_id": result.order_id,
        "order_query_status": "succeeded",
        "order_query_result": result.model_dump(mode="json"),
        "order_query_error_code": None,
        "order_query_error_kind": None,
        "order_query_error_action": None,
        "order_query_error_message": None,
        "order_query_retryable": None,
        "order_query_error_status_code": None,
        "final_answer": build_order_query_success_answer(result),
        "node_history": ["query_order"],
    }


def extract_ticket_fields_node(
    state: TicketAgentState,
    *,
    extractor: TicketFieldExtractor | None = None,
) -> TicketAgentState:
    previous_fields = (
        state.get("ticket_fields") if has_active_ticket_field_collection(state) else None
    )
    if extractor is None:
        latest_fields = extract_ticket_fields(state)
        extraction_source: TicketFieldExtractionSource = "rule_based"
    else:
        latest_fields = extractor.extract_fields(state)
        extraction_source = get_ticket_field_extraction_source(extractor)
    fields = merge_ticket_fields(previous_fields, latest_fields)

    missing_fields = find_missing_ticket_fields(fields)

    return {
        "ticket_fields": fields,
        "missing_ticket_fields": missing_fields,
        "ticket_fields_complete": not missing_fields,
        "ticket_field_extraction_source": extraction_source,
        "final_answer": _build_ticket_fields_extraction_answer(missing_fields),
        # 进入工单创建链：清除残留的退款/取消活动标志，防止确认路由把工单
        # 误路由到 execute_refund_request / execute_cancel_request 而真实执行
        # 退款/取消（worker 包装函数 _extract_ticket_fields_reset_flags 同语义）。
        "refund_request_active": False,
        "cancel_request_active": False,
        "node_history": ["extract_ticket_fields"],
    }


def ask_missing_ticket_fields_node(state: TicketAgentState) -> TicketAgentState:
    missing_fields = list(state.get("missing_ticket_fields", []))
    question = build_missing_ticket_fields_question(missing_fields)

    return {
        "missing_ticket_field_question": question,
        "missing_ticket_field_question_fields": missing_fields,
        "final_answer": question,
        "node_history": ["ask_missing_ticket_fields"],
    }


def request_ticket_confirmation_node(state: TicketAgentState) -> TicketAgentState:
    fields = state.get("ticket_fields")
    if fields is None:
        message = "当前还没有可确认的工单字段，请先补充问题信息。"
        return {
            "ticket_confirmation_required": False,
            "ticket_confirmation_message": message,
            "final_answer": message,
            "node_history": ["request_ticket_confirmation"],
        }

    pending_confirmation = build_pending_ticket_confirmation(fields)

    return {
        "ticket_confirmation_required": True,
        "ticket_confirmation_correction_requested": False,
        "ticket_confirmation_message": pending_confirmation["message"],
        "pending_ticket_confirmation": pending_confirmation,
        "final_answer": pending_confirmation["message"],
        "node_history": ["request_ticket_confirmation"],
    }


def request_ticket_confirmation_interrupt_node(
    state: TicketAgentState,
) -> TicketAgentState:
    fields = state.get("ticket_fields")
    if fields is None:
        message = "当前还没有可确认的工单字段，请先补充问题信息。"
        return {
            "ticket_confirmation_required": False,
            "ticket_confirmation_message": message,
            "final_answer": message,
            "node_history": ["request_ticket_confirmation"],
        }

    pending_confirmation = build_pending_ticket_confirmation(fields)
    # The refund *execution* draft sets refund_request_active on the worker
    # state; refund-typed tickets through the ordinary ticket flow do not.  The
    # draft fields alone are identical for both paths, so the flag is written
    # into the interrupt payload here (worker scope) where it is always
    # visible — even inside the multi-agent supervisor whose top-level state
    # never receives the worker flag.  The same applies to cancel executions
    # via cancel_request_active / is_cancel_execution.
    resume_value = interrupt(
        build_ticket_confirmation_interrupt_payload(
            pending_confirmation,
            is_refund_execution=state.get("refund_request_active") is True,
            is_cancel_execution=state.get("cancel_request_active") is True,
        )
    )
    corrected_fields = get_ticket_confirmation_resume_corrected_fields(resume_value)
    if corrected_fields is not None:
        missing_fields = find_missing_ticket_fields(corrected_fields)
        if missing_fields:
            message = build_missing_ticket_fields_question(missing_fields)
            return {
                "ticket_fields": corrected_fields,
                "missing_ticket_fields": missing_fields,
                "ticket_fields_complete": False,
                "ticket_confirmation_required": False,
                "ticket_confirmation_correction_requested": False,
                "pending_ticket_confirmation": None,
                "final_answer": message,
                "node_history": ["request_ticket_confirmation"],
            }
        return {
            "ticket_fields": corrected_fields,
            "missing_ticket_fields": [],
            "ticket_fields_complete": True,
            "ticket_confirmation_required": True,
            "ticket_confirmation_approved": False,
            "ticket_confirmation_correction_requested": True,
            "pending_ticket_confirmation": None,
            "final_answer": "工单草稿已更新，请再次确认后再创建。",
            "node_history": ["request_ticket_confirmation"],
        }
    approved = is_ticket_confirmation_resume_approved(resume_value)
    is_refund = state.get("refund_request_active") is True
    is_cancel = state.get("cancel_request_active") is True

    update: TicketAgentState = {
        "ticket_confirmation_required": True,
        "ticket_confirmation_approved": approved,
        "ticket_confirmation_correction_requested": False,
        "ticket_confirmation_message": pending_confirmation["message"],
        "pending_ticket_confirmation": pending_confirmation,
        "final_answer": (
            (
                (
                    "用户已确认取消订单，正在继续执行。"
                    if is_cancel
                    else "用户已确认退款申请，正在继续执行。"
                )
                if is_refund or is_cancel
                else "用户已确认创建工单，正在继续执行。"
            )
            if approved
            else (
                (
                    CANCEL_CONFIRMATION_REJECTED_MESSAGE
                    if is_cancel
                    else REFUND_CONFIRMATION_REJECTED_MESSAGE
                )
                if is_refund or is_cancel
                else TICKET_CONFIRMATION_REJECTED_MESSAGE
            )
        ),
        "node_history": ["request_ticket_confirmation"],
    }
    if not approved:
        # 退款/取消流程被拒绝即流程终止：清除 refund_request_active 与
        # cancel_request_active，防止同一线程后续改口创建工单时被
        # route_by_ticket_confirmation 误路由到 execute_refund_request /
        # execute_cancel_request 而真实执行退款/取消。
        update["refund_request_active"] = False
        update["cancel_request_active"] = False
    actor_id = get_ticket_confirmation_resume_actor_id(resume_value)
    if actor_id is not None:
        update["ticket_actor_id"] = actor_id
    return update


def create_ticket_node(
    state: TicketAgentState,
    creator: TicketCreator | None = None,
) -> TicketAgentState:
    if state.get("ticket_confirmation_approved") is not True:
        message = "创建工单前需要先得到用户确认。"
        safety_state = build_ticket_write_safety_state(
            status="confirmation_required",
        )
        logger.info(
            "ticket_agent_create_ticket_blocked code=%s tool_name=%s safety_status=%s",
            "TICKET_CONFIRMATION_REQUIRED",
            safety_state["ticket_tool_name"],
            safety_state["ticket_write_safety_status"],
        )
        return {
            "ticket_creation_status": "blocked",
            "ticket_creation_error_code": "TICKET_CONFIRMATION_REQUIRED",
            "ticket_creation_error_message": message,
            **safety_state,
            "final_answer": message,
            "node_history": ["create_ticket"],
        }

    fields = _get_confirmed_ticket_fields(state)
    if fields is None:
        message = "没有找到可创建工单的确认字段，请重新整理工单信息。"
        logger.warning("ticket_agent_create_ticket_failed code=%s", "TICKET_FIELDS_NOT_FOUND")
        update = build_ticket_creation_failure_state(
            code="TICKET_FIELDS_NOT_FOUND",
            message=message,
        )
        update.update(
            build_ticket_write_safety_state(status="missing_confirmed_fields")
        )
        return update

    actor_id = state.get("ticket_actor_id") or DEFAULT_TICKET_ACTOR_ID
    idempotency_key = _get_ticket_creation_idempotency_key(state, fields)

    try:
        tool_definition = authorize_tool_call(
            CREATE_TICKET_TOOL_NAME,
            user_confirmed=True,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_create_ticket_failed code=%s tool_name=%s safety_status=%s",
            exc.code,
            CREATE_TICKET_TOOL_NAME,
            "tool_not_allowed",
        )
        update = build_ticket_creation_failure_state(
            code=exc.code,
            message=exc.message,
        )
        update.update(
            build_ticket_write_safety_state(
                status="tool_not_allowed",
                idempotency_key=idempotency_key,
            )
        )
        return update

    safety_state = build_ticket_write_safety_state(
        status="authorized",
        definition=tool_definition,
        idempotency_key=idempotency_key,
    )

    try:
        arguments = build_create_ticket_args_from_fields(fields, actor_id=actor_id)
        logger.info(
            (
                "ticket_agent_create_ticket_started category=%s priority=%s "
                "related_order_id=%s tool_name=%s access_level=%s "
                "requires_confirmation=%s idempotency_key=%s"
            ),
            arguments.category,
            arguments.priority,
            _safe_log_value(arguments.related_order_id),
            safety_state["ticket_tool_name"],
            safety_state["ticket_tool_access_level"],
            safety_state["ticket_tool_requires_confirmation"],
            idempotency_key,
        )
        ticket_creator = creator or create_ticket_creator()
        ticket = ticket_creator.create_ticket(
            arguments,
            idempotency_key=idempotency_key,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_create_ticket_failed code=%s error_type=%s",
            exc.code,
            type(exc).__name__,
        )
        update = build_ticket_creation_failure_state(
            code=exc.code,
            message=exc.message,
        )
        update.update(safety_state)
        return update
    except Exception as exc:
        logger.warning(
            "ticket_agent_create_ticket_failed code=%s error_type=%s",
            TICKET_CREATION_UNEXPECTED_ERROR_CODE,
            type(exc).__name__,
        )
        update = build_ticket_creation_failure_state(
            code=TICKET_CREATION_UNEXPECTED_ERROR_CODE,
            message=TICKET_CREATION_UNEXPECTED_ERROR_MESSAGE,
        )
        update.update(safety_state)
        return update

    logger.info(
        (
            "ticket_agent_create_ticket_finished status=created ticket_id=%s "
            "category=%s priority=%s tool_name=%s access_level=%s"
        ),
        ticket.ticket_id,
        ticket.category,
        ticket.priority,
        safety_state["ticket_tool_name"],
        safety_state["ticket_tool_access_level"],
    )
    return {
        "ticket_creation_args": arguments.model_dump(mode="json"),
        "ticket_creation_status": "created",
        "ticket_creation_error_code": None,
        "ticket_creation_error_message": None,
        **safety_state,
        "created_ticket": ticket.model_dump(mode="json"),
        "final_answer": (
            f"工单已创建，工单号：{ticket.ticket_id}。客服会根据工单继续处理。"
        ),
        "node_history": ["create_ticket"],
    }


def build_refund_ticket_fields(
    normalized_message: str,
    previous_fields: TicketFields | None = None,
) -> TicketFields:
    """Build ticket-shaped fields for a refund request.

    The refund flow reuses the ticket confirmation machinery, so the collected
    order_id + reason are stored in the standard TicketFields shape
    (issue_type=refund, description=reason).  Across turns the previous draft is
    merged so a follow-up like "订单号是 A1002" completes the missing order id.
    """
    previous = previous_fields if isinstance(previous_fields, dict) else {}
    lowered_message = normalized_message.casefold()
    order_id = _extract_order_id(normalized_message) or previous.get("order_id")
    description = previous.get("description") or normalized_message.strip()

    return {
        "issue_type": "refund",
        "order_id": order_id,
        "description": description,
        "user_request": "售后退款处理",
        "urgency": _infer_ticket_urgency(lowered_message, issue_type="refund"),
        "need_human_review": True,
    }


def find_missing_refund_fields(fields: TicketFields) -> list[str]:
    missing_fields: list[str] = []
    if not fields["order_id"]:
        missing_fields.append("order_id")
    if not fields["description"].strip():
        missing_fields.append("reason")
    return missing_fields


def handle_refund_request_node(state: TicketAgentState) -> TicketAgentState:
    previous_fields = (
        state.get("ticket_fields") if has_active_refund_collection(state) else None
    )
    normalized_message = state.get("normalized_message", "").strip()
    fields = build_refund_ticket_fields(normalized_message, previous_fields)
    missing_fields = find_missing_refund_fields(fields)

    return {
        "refund_request_active": True,
        "cancel_request_active": False,
        "needs_ticket": True,
        "ticket_need_source": "explicit_user_request",
        "ticket_fields": fields,
        "missing_ticket_fields": missing_fields,
        "ticket_fields_complete": not missing_fields,
        "final_answer": build_missing_ticket_fields_question(missing_fields),
        "node_history": ["handle_refund_request"],
    }


def route_by_refund_fields(state: TicketAgentState) -> TicketFieldCompletionRoute:
    if state.get("ticket_fields_complete") is True:
        return "request_confirmation"
    return "ask_missing_fields"


def build_refund_failure_state(
    *,
    code: str,
    message: str,
) -> TicketAgentState:
    update = build_ticket_agent_fallback_state(
        node_name="execute_refund_request",
        code=code,
        message=message,
    )
    update.update(
        {
            "refund_status": "failed",
            "refund_error_code": code,
            "refund_error_message": message,
            "refund_request_active": False,
        }
    )
    return update


def execute_refund_request_node(
    state: TicketAgentState,
    refund_executor: RefundExecutor | None = None,
) -> TicketAgentState:
    if state.get("ticket_confirmation_approved") is not True:
        message = REFUND_CONFIRMATION_REQUIRED_MESSAGE
        logger.info(
            "ticket_agent_refund_blocked code=%s tool_name=%s",
            "REFUND_CONFIRMATION_REQUIRED",
            REFUND_ORDER_TOOL_NAME,
        )
        return {
            "refund_status": "blocked",
            "refund_error_code": "REFUND_CONFIRMATION_REQUIRED",
            "refund_error_message": message,
            "refund_request_active": False,
            "final_answer": message,
            "node_history": ["execute_refund_request"],
        }

    fields = _get_confirmed_ticket_fields(state)
    if fields is None:
        message = REFUND_FIELDS_NOT_FOUND_MESSAGE
        logger.warning("ticket_agent_refund_failed code=%s", "REFUND_FIELDS_NOT_FOUND")
        return build_refund_failure_state(
            code="REFUND_FIELDS_NOT_FOUND",
            message=message,
        )

    order_id = fields.get("order_id")
    reason = fields.get("description") or ""
    if not order_id or not reason.strip():
        message = REFUND_FIELDS_NOT_FOUND_MESSAGE
        logger.warning(
            "ticket_agent_refund_failed code=%s",
            "REFUND_FIELDS_INCOMPLETE",
        )
        return build_refund_failure_state(
            code="REFUND_FIELDS_INCOMPLETE",
            message=message,
        )

    actor_id = state.get("ticket_actor_id") or DEFAULT_TICKET_ACTOR_ID
    idempotency_key = _get_ticket_creation_idempotency_key(state, fields)

    try:
        tool_definition = authorize_tool_call(
            REFUND_ORDER_TOOL_NAME,
            user_confirmed=True,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_refund_failed code=%s tool_name=%s",
            exc.code,
            REFUND_ORDER_TOOL_NAME,
        )
        return build_refund_failure_state(code=exc.code, message=exc.message)

    try:
        arguments = RefundOrderArgs(
            order_id=order_id,
            reason=reason[:REFUND_REASON_MAX_LENGTH],
            requester_id=actor_id,
        )
        logger.info(
            (
                "ticket_agent_refund_started order_id=%s tool_name=%s "
                "access_level=%s requires_confirmation=%s idempotency_key=%s"
            ),
            arguments.order_id,
            REFUND_ORDER_TOOL_NAME,
            tool_definition.access_level.value,
            tool_definition.requires_confirmation,
            idempotency_key,
        )
        refund_executor = refund_executor or create_refund_executor()
        result = refund_executor.refund_order(
            arguments,
            idempotency_key=idempotency_key,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_refund_failed code=%s error_type=%s",
            exc.code,
            type(exc).__name__,
        )
        return build_refund_failure_state(code=exc.code, message=exc.message)
    except Exception as exc:
        logger.warning(
            "ticket_agent_refund_failed code=%s error_type=%s",
            REFUND_UNEXPECTED_ERROR_CODE,
            type(exc).__name__,
        )
        return build_refund_failure_state(
            code=REFUND_UNEXPECTED_ERROR_CODE,
            message=REFUND_UNEXPECTED_ERROR_MESSAGE,
        )

    logger.info(
        "ticket_agent_refund_finished status=succeeded order_id=%s",
        arguments.order_id,
    )
    return {
        "refund_status": "succeeded",
        "refund_error_code": None,
        "refund_error_message": None,
        "refund_result": dict(result),
        # 退款执行完成即流程终止：清除 refund_request_active，避免同一线程
        # 后续对话被 route_by_ticket_confirmation 误路由回退款执行。
        "refund_request_active": False,
        "final_answer": (
            f"退款已申请成功，订单号 {arguments.order_id}。"
            "退款会按订单支付渠道原路退回。"
        ),
        "node_history": ["execute_refund_request"],
    }


def build_cancel_ticket_fields(
    normalized_message: str,
    previous_fields: TicketFields | None = None,
) -> TicketFields:
    """Build ticket-shaped fields for a cancel request.

    The cancel flow reuses the ticket confirmation machinery, so the collected
    order_id + reason are stored in the standard TicketFields shape
    (issue_type=cancel, description=reason).  Across turns the previous draft is
    merged so a follow-up like "订单号是 A1002" completes the missing order id.
    """
    previous = previous_fields if isinstance(previous_fields, dict) else {}
    lowered_message = normalized_message.casefold()
    order_id = _extract_order_id(normalized_message) or previous.get("order_id")
    description = previous.get("description") or normalized_message.strip()

    return {
        "issue_type": "cancel",
        "order_id": order_id,
        "description": description,
        "user_request": "订单取消处理",
        "urgency": _infer_ticket_urgency(lowered_message, issue_type="cancel"),
        "need_human_review": True,
    }


def find_missing_cancel_fields(fields: TicketFields) -> list[str]:
    missing_fields: list[str] = []
    if not fields["order_id"]:
        missing_fields.append("order_id")
    if not fields["description"].strip():
        missing_fields.append("reason")
    return missing_fields


def handle_cancel_request_node(state: TicketAgentState) -> TicketAgentState:
    previous_fields = (
        state.get("ticket_fields") if has_active_cancel_collection(state) else None
    )
    normalized_message = state.get("normalized_message", "").strip()
    fields = build_cancel_ticket_fields(normalized_message, previous_fields)
    missing_fields = find_missing_cancel_fields(fields)

    return {
        "cancel_request_active": True,
        "refund_request_active": False,
        "needs_ticket": True,
        "ticket_need_source": "explicit_user_request",
        "ticket_fields": fields,
        "missing_ticket_fields": missing_fields,
        "ticket_fields_complete": not missing_fields,
        "final_answer": build_missing_ticket_fields_question(missing_fields),
        "node_history": ["handle_cancel_request"],
    }


def route_by_cancel_fields(state: TicketAgentState) -> TicketFieldCompletionRoute:
    if state.get("ticket_fields_complete") is True:
        return "request_confirmation"
    return "ask_missing_fields"


def build_cancel_failure_state(
    *,
    code: str,
    message: str,
) -> TicketAgentState:
    update = build_ticket_agent_fallback_state(
        node_name="execute_cancel_request",
        code=code,
        message=message,
    )
    update.update(
        {
            "cancel_status": "failed",
            "cancel_error_code": code,
            "cancel_error_message": message,
            "cancel_request_active": False,
        }
    )
    return update


def execute_cancel_request_node(
    state: TicketAgentState,
    cancel_executor: CancelExecutor | None = None,
) -> TicketAgentState:
    if state.get("ticket_confirmation_approved") is not True:
        message = CANCEL_CONFIRMATION_REQUIRED_MESSAGE
        logger.info(
            "ticket_agent_cancel_blocked code=%s tool_name=%s",
            "CANCEL_CONFIRMATION_REQUIRED",
            CANCEL_ORDER_TOOL_NAME,
        )
        return {
            "cancel_status": "blocked",
            "cancel_error_code": "CANCEL_CONFIRMATION_REQUIRED",
            "cancel_error_message": message,
            "cancel_request_active": False,
            "final_answer": message,
            "node_history": ["execute_cancel_request"],
        }

    fields = _get_confirmed_ticket_fields(state)
    if fields is None:
        message = CANCEL_FIELDS_NOT_FOUND_MESSAGE
        logger.warning("ticket_agent_cancel_failed code=%s", "CANCEL_FIELDS_NOT_FOUND")
        return build_cancel_failure_state(
            code="CANCEL_FIELDS_NOT_FOUND",
            message=message,
        )

    order_id = fields.get("order_id")
    reason = fields.get("description") or ""
    if not order_id or not reason.strip():
        message = CANCEL_FIELDS_NOT_FOUND_MESSAGE
        logger.warning(
            "ticket_agent_cancel_failed code=%s",
            "CANCEL_FIELDS_INCOMPLETE",
        )
        return build_cancel_failure_state(
            code="CANCEL_FIELDS_INCOMPLETE",
            message=message,
        )

    actor_id = state.get("ticket_actor_id") or DEFAULT_TICKET_ACTOR_ID
    idempotency_key = _get_ticket_creation_idempotency_key(state, fields)

    try:
        tool_definition = authorize_tool_call(
            CANCEL_ORDER_TOOL_NAME,
            user_confirmed=True,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_cancel_failed code=%s tool_name=%s",
            exc.code,
            CANCEL_ORDER_TOOL_NAME,
        )
        return build_cancel_failure_state(code=exc.code, message=exc.message)

    try:
        arguments = CancelOrderArgs(
            order_id=order_id,
            reason=reason[:CANCEL_REASON_MAX_LENGTH],
            requester_id=actor_id,
        )
        logger.info(
            (
                "ticket_agent_cancel_started order_id=%s tool_name=%s "
                "access_level=%s requires_confirmation=%s idempotency_key=%s"
            ),
            arguments.order_id,
            CANCEL_ORDER_TOOL_NAME,
            tool_definition.access_level.value,
            tool_definition.requires_confirmation,
            idempotency_key,
        )
        cancel_executor = cancel_executor or create_cancel_executor()
        result = cancel_executor.cancel_order(
            arguments,
            idempotency_key=idempotency_key,
        )
    except AppException as exc:
        logger.warning(
            "ticket_agent_cancel_failed code=%s error_type=%s",
            exc.code,
            type(exc).__name__,
        )
        return build_cancel_failure_state(code=exc.code, message=exc.message)
    except Exception as exc:
        logger.warning(
            "ticket_agent_cancel_failed code=%s error_type=%s",
            CANCEL_UNEXPECTED_ERROR_CODE,
            type(exc).__name__,
        )
        return build_cancel_failure_state(
            code=CANCEL_UNEXPECTED_ERROR_CODE,
            message=CANCEL_UNEXPECTED_ERROR_MESSAGE,
        )

    logger.info(
        "ticket_agent_cancel_finished status=succeeded order_id=%s",
        arguments.order_id,
    )
    return {
        "cancel_status": "succeeded",
        "cancel_error_code": None,
        "cancel_error_message": None,
        "cancel_result": dict(result),
        # 取消执行完成即流程终止：清除 cancel_request_active，避免同一线程
        # 后续对话被 route_by_ticket_confirmation 误路由回取消执行。
        "cancel_request_active": False,
        "final_answer": f"订单 {arguments.order_id} 已成功取消。",
        "node_history": ["execute_cancel_request"],
    }


def build_direct_answer_node(state: TicketAgentState) -> TicketAgentState:
    return {
        "final_answer": "你好，我是智能客服工单助手，可以帮你查询规则、订单和创建客服工单。",
        "node_history": ["build_direct_answer"],
    }


def build_unsupported_answer_node(state: TicketAgentState) -> TicketAgentState:
    return {
        "final_answer": "这个请求超出当前智能客服工单助手 v1 的处理范围。",
        "node_history": ["build_unsupported_answer"],
    }


def ask_clarifying_question_node(state: TicketAgentState) -> TicketAgentState:
    return {
        "final_answer": "我还不能确定你要处理的问题，请补充订单号、问题类型或具体诉求。",
        "node_history": ["ask_clarifying_question"],
    }


def build_ticket_agent_graph(
    ticket_creator: TicketCreator | None = None,
    *,
    policy_rag_service: PolicyRagService | None = None,
    order_query_executor: OrderQueryExecutor | None = None,
    intent_classifier: TicketIntentClassifier | None = None,
    field_extractor: TicketFieldExtractor | None = None,
    emotion_classifier: CustomerEmotionClassifier | None = None,
    refund_executor: RefundExecutor | None = None,
    cancel_executor: CancelExecutor | None = None,
    checkpointer: Any | None = None,
    interrupt_confirmation: bool = False,
):
    builder = StateGraph(TicketAgentState)

    builder.add_node("normalize_user_input", normalize_user_input_node)
    builder.add_node(
        "emotion_recognition",
        lambda state: emotion_recognition_node(
            state,
            classifier=emotion_classifier or RuleBasedEmotionClassifier(),
        ),
    )
    builder.add_node(
        "classify_intent",
        lambda state: classify_intent_node(state, classifier=intent_classifier),
    )
    builder.add_node(
        "retrieve_policy",
        lambda state: retrieve_policy_node(state, service=policy_rag_service),
    )
    builder.add_node("decide_ticket_need", decide_ticket_need_node)
    builder.add_node(
        "query_order",
        lambda state: query_order_node(
            state,
            order_query_executor=order_query_executor,
        ),
    )
    builder.add_node(
        "extract_ticket_fields",
        lambda state: extract_ticket_fields_node(state, extractor=field_extractor),
    )
    builder.add_node("handle_refund_request", handle_refund_request_node)
    builder.add_node("handle_cancel_request", handle_cancel_request_node)
    builder.add_node("ask_missing_ticket_fields", ask_missing_ticket_fields_node)
    builder.add_node(
        "request_ticket_confirmation",
        (
            request_ticket_confirmation_interrupt_node
            if interrupt_confirmation
            else request_ticket_confirmation_node
        ),
    )
    builder.add_node(
        "create_ticket",
        lambda state: create_ticket_node(state, creator=ticket_creator),
    )
    builder.add_node(
        "execute_refund_request",
        lambda state: execute_refund_request_node(
            state,
            refund_executor=refund_executor,
        ),
    )
    builder.add_node(
        "execute_cancel_request",
        lambda state: execute_cancel_request_node(
            state,
            cancel_executor=cancel_executor,
        ),
    )
    builder.add_node("build_direct_answer", build_direct_answer_node)
    builder.add_node("build_unsupported_answer", build_unsupported_answer_node)
    builder.add_node("ask_clarifying_question", ask_clarifying_question_node)
    builder.add_node(
        "build_emotion_handoff_answer",
        build_emotion_handoff_answer_node,
    )

    for start_node, end_node in TICKET_AGENT_FIXED_EDGES:
        builder.add_edge(start_node, end_node)

    builder.add_conditional_edges(
        "emotion_recognition",
        route_by_emotion,
        EMOTION_ROUTES,
    )
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        TICKET_AGENT_INTENT_ROUTES,
    )
    builder.add_conditional_edges(
        "decide_ticket_need",
        route_by_ticket_need,
        TICKET_AGENT_TICKET_NEED_ROUTES,
    )
    builder.add_conditional_edges(
        "extract_ticket_fields",
        route_by_ticket_fields_complete,
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_conditional_edges(
        "handle_refund_request",
        route_by_refund_fields,
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_conditional_edges(
        "handle_cancel_request",
        route_by_cancel_fields,
        TICKET_AGENT_FIELD_COMPLETION_ROUTES,
    )
    builder.add_conditional_edges(
        "request_ticket_confirmation",
        route_by_ticket_confirmation,
        TICKET_AGENT_CONFIRMATION_ROUTES,
    )

    return builder.compile(checkpointer=checkpointer)


def build_ticket_agent_graph_for_model_mode(
    ticket_creator: TicketCreator | None = None,
    *,
    policy_rag_service: PolicyRagService | None = None,
    order_query_executor: OrderQueryExecutor | None = None,
    mode: TicketAgentModelMode | None = None,
    settings: Settings | None = None,
    client: Any | None = None,
    intent_prompt_spec: TicketAgentPromptSpec = TICKET_INTENT_CLASSIFICATION_PROMPT,
    field_prompt_spec: TicketAgentPromptSpec = TICKET_FIELD_EXTRACTION_PROMPT,
    enable_model_output_fallback: bool = False,
    refund_executor: RefundExecutor | None = None,
    cancel_executor: CancelExecutor | None = None,
    emotion_classifier: CustomerEmotionClassifier | None = None,
    checkpointer: Any | None = None,
    interrupt_confirmation: bool = False,
):
    dependencies = create_ticket_agent_model_dependencies(
        mode,
        settings=settings,
        client=client,
        intent_prompt_spec=intent_prompt_spec,
        field_prompt_spec=field_prompt_spec,
        enable_model_output_fallback=enable_model_output_fallback,
    )

    return build_ticket_agent_graph(
        ticket_creator=ticket_creator,
        policy_rag_service=policy_rag_service,
        order_query_executor=order_query_executor,
        intent_classifier=dependencies["intent_classifier"],
        field_extractor=dependencies["field_extractor"],
        emotion_classifier=emotion_classifier or dependencies.get("emotion_classifier"),
        refund_executor=refund_executor,
        cancel_executor=cancel_executor,
        checkpointer=checkpointer,
        interrupt_confirmation=interrupt_confirmation,
    )


ticket_agent_graph = build_ticket_agent_graph()


def build_checkpointed_ticket_agent_graph(
    ticket_creator: TicketCreator | None = None,
    *,
    policy_rag_service: PolicyRagService | None = None,
    order_query_executor: OrderQueryExecutor | None = None,
    intent_classifier: TicketIntentClassifier | None = None,
    field_extractor: TicketFieldExtractor | None = None,
    refund_executor: RefundExecutor | None = None,
    cancel_executor: CancelExecutor | None = None,
):
    return build_ticket_agent_graph(
        ticket_creator=ticket_creator,
        policy_rag_service=policy_rag_service,
        order_query_executor=order_query_executor,
        intent_classifier=intent_classifier,
        field_extractor=field_extractor,
        refund_executor=refund_executor,
        cancel_executor=cancel_executor,
        checkpointer=MemorySaver(),
    )


def build_interrupting_ticket_agent_graph(
    ticket_creator: TicketCreator | None = None,
    *,
    policy_rag_service: PolicyRagService | None = None,
    order_query_executor: OrderQueryExecutor | None = None,
    intent_classifier: TicketIntentClassifier | None = None,
    field_extractor: TicketFieldExtractor | None = None,
    refund_executor: RefundExecutor | None = None,
    cancel_executor: CancelExecutor | None = None,
):
    return build_ticket_agent_graph(
        ticket_creator=ticket_creator,
        policy_rag_service=policy_rag_service,
        order_query_executor=order_query_executor,
        intent_classifier=intent_classifier,
        field_extractor=field_extractor,
        refund_executor=refund_executor,
        cancel_executor=cancel_executor,
        checkpointer=MemorySaver(),
        interrupt_confirmation=True,
    )


def build_ticket_agent_input(
    user_message: str,
    history: list[str] | None = None,
    *,
    actor_roles: tuple[str, ...] | None = None,
    actor_tenant_id: str | None = None,
) -> TicketAgentState:
    state: TicketAgentState = {
        "user_message": user_message,
        "agent_trace_id": get_trace_id(),
        "node_history": [],
        "actor_roles": actor_roles,
        "actor_tenant_id": actor_tenant_id,
    }
    if history:
        state["history"] = history
    return state


def build_ticket_agent_thread_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": normalize_ticket_agent_thread_id(thread_id),
        }
    }


def run_ticket_agent(user_message: str) -> TicketAgentState:
    start_time = perf_counter()
    log_ticket_agent_run_started(
        operation="invoke",
        user_message=user_message,
    )
    try:
        result = ticket_agent_graph.invoke(build_ticket_agent_input(user_message))
    except Exception as exc:
        log_ticket_agent_run_failed(
            exc,
            operation="invoke",
            elapsed_ms=_elapsed_ms_since(start_time),
        )
        raise
    log_ticket_agent_run_finished(
        result,
        operation="invoke",
        elapsed_ms=_elapsed_ms_since(start_time),
    )
    return result


def run_ticket_agent_safely(
    user_message: str,
    *,
    graph: Any | None = None,
) -> TicketAgentState:
    selected_graph = graph or ticket_agent_graph
    start_time = perf_counter()
    log_ticket_agent_run_started(
        operation="invoke_safe",
        user_message=user_message,
    )
    try:
        result = selected_graph.invoke(build_ticket_agent_input(user_message))
    except AppException as exc:
        elapsed_ms = _elapsed_ms_since(start_time)
        log_ticket_agent_run_failed(
            exc,
            operation="invoke_safe",
            elapsed_ms=elapsed_ms,
        )
        fallback = build_ticket_agent_fallback_state(
            node_name="ticket_agent_graph",
            code=exc.code,
            message=exc.message,
        )
        log_ticket_agent_run_finished(
            fallback,
            operation="invoke_safe",
            elapsed_ms=elapsed_ms,
        )
        return fallback
    except Exception as exc:
        elapsed_ms = _elapsed_ms_since(start_time)
        log_ticket_agent_run_failed(
            exc,
            operation="invoke_safe",
            elapsed_ms=elapsed_ms,
        )
        fallback = build_ticket_agent_fallback_state(
            node_name="ticket_agent_graph",
        )
        log_ticket_agent_run_finished(
            fallback,
            operation="invoke_safe",
            elapsed_ms=elapsed_ms,
        )
        return fallback
    log_ticket_agent_run_finished(
        result,
        operation="invoke_safe",
        elapsed_ms=_elapsed_ms_since(start_time),
    )
    return result


def run_ticket_agent_in_thread(
    graph: Any,
    user_message: str,
    *,
    thread_id: str,
    actor_id: str | None = None,
    config: dict[str, Any] | None = None,
    history: list[str] | None = None,
    actor_roles: tuple[str, ...] | None = None,
    actor_tenant_id: str | None = None,
) -> TicketAgentState:
    initial_state = build_ticket_agent_input(
        user_message,
        history=history,
        actor_roles=actor_roles,
        actor_tenant_id=actor_tenant_id,
    )
    if actor_id is not None:
        initial_state["ticket_actor_id"] = actor_id

    start_time = perf_counter()
    log_ticket_agent_run_started(
        operation="invoke_thread",
        user_message=user_message,
        thread_id=thread_id,
        actor_id=actor_id,
    )
    try:
        thread_config = build_ticket_agent_thread_config(thread_id)
        if config is not None:
            # graph_config (LangSmith trace context) may override configurable
            # keys; both carry the same thread_id so the merge is value-safe.
            thread_config = {**thread_config, **config}
        result = graph.invoke(
            initial_state,
            config=thread_config,
        )
    except Exception as exc:
        log_ticket_agent_run_failed(
            exc,
            operation="invoke_thread",
            elapsed_ms=_elapsed_ms_since(start_time),
            thread_id=thread_id,
        )
        raise
    log_ticket_agent_run_finished(
        result,
        operation="invoke_thread",
        elapsed_ms=_elapsed_ms_since(start_time),
        thread_id=thread_id,
    )
    return result


def get_ticket_agent_thread_state(graph: Any, *, thread_id: str) -> TicketAgentState:
    snapshot = graph.get_state(build_ticket_agent_thread_config(thread_id))
    return dict(snapshot.values)


def build_ticket_agent_checkpoint_snapshot(
    graph: Any,
    *,
    thread_id: str,
    metadata: dict[str, Any] | None = None,
) -> TicketAgentCheckpointSnapshot:
    return TicketAgentCheckpointSnapshot.create(
        thread_id=thread_id,
        values=get_ticket_agent_thread_state(graph, thread_id=thread_id),
        metadata=metadata,
    )


def save_ticket_agent_checkpoint_snapshot(
    graph: Any,
    *,
    thread_id: str,
    store: FileTicketAgentCheckpointStore,
    metadata: dict[str, Any] | None = None,
) -> Path:
    snapshot = build_ticket_agent_checkpoint_snapshot(
        graph,
        thread_id=thread_id,
        metadata=metadata,
    )
    return store.save(snapshot)


def approve_ticket_confirmation_and_resume(
    graph: Any,
    *,
    thread_id: str,
    actor_id: str | None = None,
) -> TicketAgentState:
    config = build_ticket_agent_thread_config(thread_id)
    current_state = graph.get_state(config).values
    if current_state.get("pending_ticket_confirmation") is None:
        raise AppException(
            code="TICKET_CONFIRMATION_NOT_FOUND",
            message=TICKET_CONFIRMATION_NOT_FOUND_MESSAGE,
            status_code=409,
        )

    approved_update: TicketAgentState = {"ticket_confirmation_approved": True}
    if actor_id is not None:
        approved_update["ticket_actor_id"] = actor_id

    graph.update_state(
        config,
        approved_update,
        as_node="request_ticket_confirmation",
    )
    return graph.invoke(None, config=config)


def get_ticket_confirmation_interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        raise AppException(
            code="TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND",
            message=TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND_MESSAGE,
            status_code=409,
        )

    interrupt_value = interrupts[0].value
    if (
        not isinstance(interrupt_value, dict)
        or interrupt_value.get("kind") != TICKET_CONFIRMATION_INTERRUPT_KIND
    ):
        raise AppException(
            code="TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND",
            message=TICKET_CONFIRMATION_INTERRUPT_NOT_FOUND_MESSAGE,
            status_code=409,
        )

    return interrupt_value


def resume_ticket_confirmation_interrupt(
    graph: Any,
    *,
    thread_id: str,
    approved: bool,
    actor_id: str | None = None,
    corrected_fields: TicketFields | None = None,
) -> TicketAgentState:
    resume_payload: dict[str, Any] = {"approved": approved}
    if actor_id is not None:
        resume_payload["actor_id"] = actor_id
    if corrected_fields is not None:
        resume_payload["corrected_ticket_fields"] = corrected_fields

    start_time = perf_counter()
    log_ticket_agent_run_started(
        operation="resume_interrupt",
        thread_id=thread_id,
        actor_id=actor_id,
    )
    try:
        result = graph.invoke(
            Command(resume=resume_payload),
            config=build_ticket_agent_thread_config(thread_id),
        )
    except Exception as exc:
        log_ticket_agent_run_failed(
            exc,
            operation="resume_interrupt",
            elapsed_ms=_elapsed_ms_since(start_time),
            thread_id=thread_id,
        )
        raise
    log_ticket_agent_run_finished(
        result,
        operation="resume_interrupt",
        elapsed_ms=_elapsed_ms_since(start_time),
        thread_id=thread_id,
    )
    return result


def resume_ticket_confirmation_interrupt_safely(
    graph: Any,
    *,
    thread_id: str,
    approved: bool,
    actor_id: str | None = None,
) -> TicketAgentState:
    try:
        return resume_ticket_confirmation_interrupt(
            graph,
            thread_id=thread_id,
            approved=approved,
            actor_id=actor_id,
        )
    except AppException as exc:
        return build_ticket_agent_fallback_state(
            node_name="resume_ticket_confirmation_interrupt",
            code=exc.code,
            message=exc.message,
        )
    except ValueError as exc:
        return build_ticket_agent_fallback_state(
            node_name="resume_ticket_confirmation_interrupt",
            code=TICKET_THREAD_ID_INVALID_ERROR_CODE,
            message=str(exc),
        )
    except Exception:
        return build_ticket_agent_fallback_state(
            node_name="resume_ticket_confirmation_interrupt",
        )


def stream_ticket_agent_updates(user_message: str) -> list[TicketAgentStreamPart]:
    return list(
        ticket_agent_graph.stream(
            build_ticket_agent_input(user_message),
            stream_mode="updates",
            version="v2",
        )
    )


def create_policy_rag_service() -> PolicyRagService:
    """⚠️ deprecated: 仅测试/兜底使用（FakePolicyRagService）。

    生产对话链路由 ConsoleAgentService 注入 ProductionPolicyRagService（真实 RAG）；
    此函数仅在被显式注入 None 时兜底，新代码不应依赖。
    """
    return FakePolicyRagService()


def create_ticket_creator() -> TicketCreator:
    return JavaTicketClient.from_settings(get_settings())


def create_refund_executor() -> RefundExecutor:
    from app.agents.mcp_tool_adapters import create_mcp_refund_executor

    return create_mcp_refund_executor()


def create_cancel_executor() -> CancelExecutor:
    from app.agents.mcp_tool_adapters import create_mcp_cancel_executor

    return create_mcp_cancel_executor()


class JavaRefundExecutor:
    """RefundExecutor backed directly by the Java business service (direct-Java mode).

    The direct path is idempotency-keyed at the Java service, so the MCP
    confirmation store is not involved here (mirrors JavaTicketClient for
    create_ticket).  Identity flows through the business context headers set by
    the console agent service for the authenticated actor.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def refund_order(
        self,
        arguments: RefundOrderArgs,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        result = self._client.refund_order(
            arguments.order_id,
            arguments.reason,
            idempotency_key=idempotency_key,
        )
        return dict(result)


def create_java_refund_executor() -> RefundExecutor:
    from app.services.java_order_client import JavaOrderClient

    return JavaRefundExecutor(JavaOrderClient.from_settings(get_settings()))


class JavaCancelExecutor:
    """CancelExecutor backed directly by the Java business service (direct-Java mode).

    The direct path is idempotency-keyed at the Java service, so the MCP
    confirmation store is not involved here (mirrors JavaTicketClient for
    create_ticket).  Identity flows through the business context headers set by
    the console agent service for the authenticated actor.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def cancel_order(
        self,
        arguments: CancelOrderArgs,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        result = self._client.cancel_order(
            arguments.order_id,
            arguments.reason,
            idempotency_key=idempotency_key,
        )
        return dict(result)


def create_java_cancel_executor() -> CancelExecutor:
    from app.services.java_order_client import JavaOrderClient

    return JavaCancelExecutor(JavaOrderClient.from_settings(get_settings()))


def _make_fake_retrieved_chunk(
    *,
    chunk_id: str,
    content: str,
    source: str,
    title: str,
    section: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=f"fake-{chunk_id}",
        chunk_id=chunk_id,
        content=content,
        metadata={
            "source": source,
            "title": title,
            "section": section,
            "doc_type": "policy",
            "permission_group": "customer_service",
        },
        score=0.91,
    )


def _elapsed_ms_since(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000


def _safe_log_value(value: object | None) -> str:
    if value is None:
        return TICKET_AGENT_LOG_VALUE_EMPTY
    text = str(value).strip()
    return text or TICKET_AGENT_LOG_VALUE_EMPTY


def _contains_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in message for keyword in keywords)


def _extract_order_id(message: str) -> str | None:
    match = ORDER_ID_PATTERN.search(message)
    if match is not None:
        return match.group(1).strip()

    fallback_match = FALLBACK_ORDER_ID_PATTERN.search(message)
    if fallback_match is not None:
        return fallback_match.group(1).strip()

    return None


def _infer_ticket_issue_type(
    lowered_message: str,
    *,
    ticket_need_source: TicketNeedSource | None,
    rag_answer_status: str | None,
) -> TicketIssueType:
    if _contains_any(lowered_message, COMPLAINT_ISSUE_KEYWORDS):
        return "complaint"
    if _contains_any(lowered_message, LOGISTICS_ISSUE_KEYWORDS):
        return "logistics"
    if _contains_any(lowered_message, REFUND_ISSUE_KEYWORDS):
        return "refund"
    if ticket_need_source == "rag_no_context" or rag_answer_status == "no_context":
        return "policy_gap"
    return "unknown"


def _infer_ticket_user_request(
    lowered_message: str,
    *,
    issue_type: TicketIssueType,
    ticket_need_source: TicketNeedSource | None,
) -> str:
    if "投诉" in lowered_message:
        return "投诉处理"
    if "创建工单" in lowered_message or "工单" in lowered_message:
        return "创建工单"
    if "人工" in lowered_message or "处理" in lowered_message:
        return "人工处理"
    if issue_type == "policy_gap" or ticket_need_source == "rag_no_context":
        return "补充或人工解释知识库未覆盖问题"
    if issue_type == "refund":
        return "售后退款处理"
    if issue_type == "logistics":
        return "物流问题处理"
    if issue_type == "complaint":
        return "投诉处理"
    return ""


def _infer_ticket_urgency(
    lowered_message: str,
    *,
    issue_type: TicketIssueType,
) -> TicketUrgencyLevel:
    if _contains_any(lowered_message, HIGH_URGENCY_KEYWORDS):
        return "high"
    if issue_type == "policy_gap":
        return "normal"
    return "normal"


def _build_ticket_description(
    normalized_message: str,
    *,
    ticket_need_source: TicketNeedSource | None,
) -> str:
    if ticket_need_source == "rag_no_context":
        return f"用户问题：{normalized_message}；知识库未找到足够资料。"
    return normalized_message


def _build_ticket_fields_extraction_answer(missing_fields: list[str]) -> str:
    if missing_fields:
        return (
            "已进入工单流程，并抽取了部分工单字段；仍缺少："
            f"{'、'.join(missing_fields)}。后续课程会学习如何追问缺失字段。"
        )
    return "已进入工单流程，并抽取了初步工单字段；后续课程会学习如何请求用户确认。"


def _build_ticket_creation_title(fields: TicketFields) -> str:
    issue_type_label = TICKET_ISSUE_TYPE_LABELS[fields["issue_type"]]
    order_part = f"订单 {fields['order_id']}" if fields["order_id"] else "无订单号"
    title = f"{issue_type_label}：{order_part}，{fields['user_request']}"
    return title[:200]


def _get_confirmed_ticket_fields(state: TicketAgentState) -> TicketFields | None:
    pending_confirmation = state.get("pending_ticket_confirmation")
    if pending_confirmation is not None:
        return pending_confirmation["ticket_fields"]
    return state.get("ticket_fields")


def _get_ticket_creation_idempotency_key(
    state: TicketAgentState,
    fields: TicketFields,
) -> str:
    pending_confirmation = state.get("pending_ticket_confirmation")
    if pending_confirmation is not None:
        return pending_confirmation["confirmation_id"]
    return build_ticket_confirmation_id(fields)
