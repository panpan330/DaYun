import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from app.core.config import Settings
from app.core.cost_control import (
    LLMCostControlDecision,
    build_llm_cost_control_decision,
)
from app.core.exceptions import AppException
from app.core.llm_logging_safety import build_safe_llm_log_payload
from app.core.llm_retry import (
    LLMRetryDecision,
    build_llm_retry_decision,
    extract_retry_after_seconds,
)
from app.core.llm_timeout import (
    LLMTimeoutBudgetDecision,
    build_llm_timeout_budget_decision,
)
from app.core.model_fallback import (
    LLMFallbackDecision,
    build_llm_fallback_decision,
    disable_fallback_by_cost_control,
)
from app.core.model_routing import LLMModelRouteDecision, route_llm_model
from app.core.token_usage import TokenCostRecord, TokenPricing, build_token_cost_record
from app.schemas.chat import ChatMessage
from app.services.llm_client import create_openai_compatible_client
from app.services.message_builder import (
    build_multi_turn_messages,
    serialize_chat_messages,
)
from app.services.prompt_builder import PromptParts, build_clear_user_prompt


logger = logging.getLogger(__name__)

DEFAULT_CHAT_CONSTRAINTS = (
    "用中文回答",
    "回答适合刚开始学习 AI 应用开发的人",
    "解释概念时先讲人话，再补充术语",
    "不要编造不确定的信息",
)
DEFAULT_CHAT_OUTPUT_FORMAT = "先直接回答用户问题，再在需要时补充关键要点。"
DEFAULT_CHAT_FAILURE_POLICY = "如果不确定，请明确说不确定，并说明需要查官方文档。"


@dataclass(frozen=True)
class LLMTokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def build_chat_prompt(user_message: str) -> str:
    return build_clear_user_prompt(
        PromptParts(
            task=user_message,
            constraints=DEFAULT_CHAT_CONSTRAINTS,
            output_format=DEFAULT_CHAT_OUTPUT_FORMAT,
            failure_policy=DEFAULT_CHAT_FAILURE_POLICY,
        )
    )


def build_chat_messages(
    user_message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> list[ChatMessage]:
    return build_multi_turn_messages(
        build_chat_prompt(user_message),
        history=history,
    )


def extract_first_reply(completion: Any) -> str:
    try:
        reply = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise AppException(
            code="LLM_BAD_RESPONSE",
            message="模型返回格式异常",
            status_code=502,
        ) from exc

    if not isinstance(reply, str) or not reply.strip():
        raise AppException(
            code="LLM_EMPTY_RESPONSE",
            message="模型返回了空内容",
            status_code=502,
        )
    return reply.strip()


def _get_usage_value(usage: Any, field_name: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(field_name)
    else:
        value = getattr(usage, field_name, None)

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def extract_token_usage(completion: Any) -> LLMTokenUsage:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return LLMTokenUsage()

    return LLMTokenUsage(
        prompt_tokens=_get_usage_value(usage, "prompt_tokens"),
        completion_tokens=_get_usage_value(usage, "completion_tokens"),
        total_tokens=_get_usage_value(usage, "total_tokens"),
    )


def _get_first_choice(response_part: Any) -> Any | None:
    if isinstance(response_part, dict):
        choices = response_part.get("choices")
    else:
        choices = getattr(response_part, "choices", None)

    try:
        return choices[0]
    except (IndexError, TypeError):
        return None


def extract_stream_delta_content(chunk: Any) -> str | None:
    choice = _get_first_choice(chunk)
    if choice is None:
        return None

    if isinstance(choice, dict):
        delta = choice.get("delta")
    else:
        delta = getattr(choice, "delta", None)

    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)

    if not isinstance(content, str) or content == "":
        return None
    return content


def has_token_usage(usage: LLMTokenUsage) -> bool:
    return any(
        value is not None
        for value in (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
    )


def map_openai_error_to_app_exception(exc: Exception) -> AppException:
    if isinstance(exc, RateLimitError):
        return AppException(
            code="LLM_RATE_LIMITED",
            message="模型服务请求过于频繁，请稍后重试。",
            status_code=429,
        )
    if isinstance(exc, APITimeoutError):
        return AppException(
            code="LLM_TIMEOUT",
            message="模型调用超时，请稍后重试。",
            status_code=504,
        )
    if isinstance(exc, AuthenticationError):
        return AppException(
            code="LLM_AUTHENTICATION_FAILED",
            message="模型服务认证失败，请检查服务端 API key 配置。",
            status_code=502,
        )
    if isinstance(exc, PermissionDeniedError):
        return AppException(
            code="LLM_PERMISSION_DENIED",
            message="模型服务拒绝访问，请检查服务端模型权限配置。",
            status_code=502,
        )
    if isinstance(exc, NotFoundError):
        return AppException(
            code="LLM_RESOURCE_NOT_FOUND",
            message="模型服务资源不存在，请检查模型名或接口地址配置。",
            status_code=502,
        )
    if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        return AppException(
            code="LLM_BAD_REQUEST",
            message="模型请求参数错误，请联系开发者检查模型调用配置。",
            status_code=502,
        )
    if isinstance(exc, InternalServerError):
        return AppException(
            code="LLM_PROVIDER_ERROR",
            message="模型服务暂时异常，请稍后重试。",
            status_code=502,
        )
    if isinstance(exc, APIConnectionError):
        return AppException(
            code="LLM_CONNECTION_ERROR",
            message="无法连接模型服务，请稍后重试。",
            status_code=502,
        )
    if isinstance(exc, APIStatusError):
        return AppException(
            code="LLM_PROVIDER_STATUS_ERROR",
            message="模型服务返回异常状态，请稍后重试。",
            status_code=502,
        )
    return AppException(
        code="LLM_CALL_FAILED",
        message="模型调用失败，请稍后重试。",
        status_code=502,
    )


class LLMChatService:
    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        *,
        sleep_func: Callable[[float], None] | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._sleep_func = sleep_func or time.sleep
        self._time_func = time_func or perf_counter

    def _now(self) -> float:
        return self._time_func()

    def _elapsed_seconds(self, start_time: float) -> float:
        return self._now() - start_time

    def _elapsed_ms(self, start_time: float) -> float:
        return self._elapsed_seconds(start_time) * 1000

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
        usage: LLMTokenUsage,
        route_decision: LLMModelRouteDecision,
    ) -> None:
        cost_record = self._build_token_cost_record("chat", usage, route_decision)
        self._record_cost(route_decision.model, usage, cost_record)
        payload = build_safe_llm_log_payload(
            operation="chat",
            outcome="success",
            provider=route_decision.provider,
            model=route_decision.model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            extra_fields={
                **route_decision.to_log_fields(),
                **cost_record.to_log_fields(),
            },
        )
        logger.info(
            (
                "llm_chat_succeeded provider=%s model=%s elapsed_ms=%.2f "
                "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
                "route_tier=%s route_reason=%s cost_status=%s "
                "estimated_cost=%s currency=%s"
            ),
            payload["llm.provider"],
            payload["llm.model"],
            payload["llm.elapsed_ms"],
            payload.get("llm.prompt_tokens"),
            payload.get("llm.completion_tokens"),
            payload.get("llm.total_tokens"),
            payload.get("llm.route_tier"),
            payload.get("llm.route_reason"),
            payload.get("llm.cost_status"),
            payload.get("llm.estimated_total_cost"),
            payload.get("llm.cost_currency"),
        )

    def _log_failure(
        self,
        app_exception: AppException,
        elapsed_ms: float,
        *,
        route_decision: LLMModelRouteDecision,
        fallback_decision: LLMFallbackDecision | None = None,
        exc_info: bool = False,
    ) -> None:
        extra_fields = dict(route_decision.to_log_fields())
        if fallback_decision is not None:
            extra_fields.update(fallback_decision.to_log_fields())

        payload = build_safe_llm_log_payload(
            operation="chat",
            outcome="failure",
            provider=route_decision.provider,
            model=route_decision.model,
            elapsed_ms=elapsed_ms,
            error_code=app_exception.code,
            status_code=app_exception.status_code,
            extra_fields=extra_fields,
        )
        logger.warning(
            (
                "llm_chat_failed code=%s provider=%s model=%s status_code=%s "
                "elapsed_ms=%.2f route_tier=%s route_reason=%s "
                "fallback_attempted=%s fallback_reason=%s"
            ),
            payload["llm.error_code"],
            payload["llm.provider"],
            payload["llm.model"],
            payload["http.status_code"],
            payload["llm.elapsed_ms"],
            payload.get("llm.route_tier"),
            payload.get("llm.route_reason"),
            payload.get("llm.fallback_attempted"),
            payload.get("llm.fallback_reason"),
            exc_info=exc_info,
        )

    def _log_stream_success(
        self,
        elapsed_ms: float,
        usage: LLMTokenUsage,
        *,
        chunk_count: int,
        content_chunk_count: int,
        route_decision: LLMModelRouteDecision,
    ) -> None:
        cost_record = self._build_token_cost_record(
            "stream_chat",
            usage,
            route_decision,
        )
        self._record_cost(route_decision.model, usage, cost_record)
        payload = build_safe_llm_log_payload(
            operation="stream_chat",
            outcome="success",
            provider=route_decision.provider,
            model=route_decision.model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            extra_fields={
                **route_decision.to_log_fields(),
                **cost_record.to_log_fields(),
                "llm.chunks": chunk_count,
                "llm.content_chunks": content_chunk_count,
            },
        )
        logger.info(
            (
                "llm_stream_chat_succeeded provider=%s model=%s elapsed_ms=%.2f "
                "chunks=%s content_chunks=%s prompt_tokens=%s "
                "completion_tokens=%s total_tokens=%s route_tier=%s "
                "route_reason=%s cost_status=%s estimated_cost=%s currency=%s"
            ),
            payload["llm.provider"],
            payload["llm.model"],
            payload["llm.elapsed_ms"],
            payload["llm.chunks"],
            payload["llm.content_chunks"],
            payload.get("llm.prompt_tokens"),
            payload.get("llm.completion_tokens"),
            payload.get("llm.total_tokens"),
            payload.get("llm.route_tier"),
            payload.get("llm.route_reason"),
            payload.get("llm.cost_status"),
            payload.get("llm.estimated_total_cost"),
            payload.get("llm.cost_currency"),
        )

    def _build_token_cost_record(
        self,
        operation: str,
        usage: LLMTokenUsage,
        route_decision: LLMModelRouteDecision,
    ) -> TokenCostRecord:
        return build_token_cost_record(
            usage,
            provider=route_decision.provider,
            model=route_decision.model,
            operation=operation,
            pricing=self._build_token_pricing(),
        )

    def _record_cost(
        self,
        model: str,
        usage: LLMTokenUsage,
        cost_record: TokenCostRecord,
    ) -> None:
        """Record this call into the in-process cost aggregator (best effort)."""
        try:
            from app.services.cost_recorder import (
                current_cost_intent,
                get_cost_recorder,
            )

            get_cost_recorder().record(
                model=model,
                intent=current_cost_intent(),
                input_tokens=usage.prompt_tokens or 0,
                output_tokens=usage.completion_tokens or 0,
                estimated_cost=float(cost_record.cost.total_cost or 0.0),
            )
        except Exception:  # pragma: no cover - cost tracking must never break chat
            logger.debug("cost_recorder_record_failed", exc_info=True)

    def _build_token_pricing(self) -> TokenPricing | None:
        if not self.settings.has_llm_token_pricing:
            return None
        return TokenPricing(
            input_cost_per_million_tokens=(
                self.settings.llm_input_cost_per_million_tokens or 0.0
            ),
            output_cost_per_million_tokens=(
                self.settings.llm_output_cost_per_million_tokens or 0.0
            ),
            currency=self.settings.resolved_llm_pricing_currency,
        )

    def _log_stream_failure(
        self,
        app_exception: AppException,
        elapsed_ms: float,
        *,
        chunk_count: int,
        content_chunk_count: int,
        route_decision: LLMModelRouteDecision,
        fallback_decision: LLMFallbackDecision | None = None,
        exc_info: bool = False,
    ) -> None:
        extra_fields = {
            **route_decision.to_log_fields(),
            "llm.chunks": chunk_count,
            "llm.content_chunks": content_chunk_count,
        }
        if fallback_decision is not None:
            extra_fields.update(fallback_decision.to_log_fields())

        payload = build_safe_llm_log_payload(
            operation="stream_chat",
            outcome="failure",
            provider=route_decision.provider,
            model=route_decision.model,
            elapsed_ms=elapsed_ms,
            error_code=app_exception.code,
            status_code=app_exception.status_code,
            extra_fields=extra_fields,
        )
        logger.warning(
            (
                "llm_stream_chat_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f chunks=%s content_chunks=%s "
                "route_tier=%s route_reason=%s fallback_attempted=%s "
                "fallback_reason=%s"
            ),
            payload["llm.error_code"],
            payload["llm.provider"],
            payload["llm.model"],
            payload["http.status_code"],
            payload["llm.elapsed_ms"],
            payload["llm.chunks"],
            payload["llm.content_chunks"],
            payload.get("llm.route_tier"),
            payload.get("llm.route_reason"),
            payload.get("llm.fallback_attempted"),
            payload.get("llm.fallback_reason"),
            exc_info=exc_info,
        )

    def _log_fallback_started(
        self,
        *,
        operation: str,
        fallback_decision: LLMFallbackDecision,
    ) -> None:
        fallback_route = fallback_decision.fallback_route
        if fallback_route is None:
            return
        logger.info(
            (
                "llm_fallback_started operation=%s primary_model=%s "
                "fallback_model=%s fallback_tier=%s primary_code=%s"
            ),
            operation,
            fallback_decision.primary_model,
            fallback_route.model,
            fallback_route.tier,
            fallback_decision.primary_error_code,
        )

    def _log_fallback_succeeded(
        self,
        *,
        operation: str,
        fallback_decision: LLMFallbackDecision,
        elapsed_ms: float,
    ) -> None:
        fallback_route = fallback_decision.fallback_route
        if fallback_route is None:
            return
        logger.info(
            (
                "llm_fallback_succeeded operation=%s primary_model=%s "
                "fallback_model=%s fallback_tier=%s primary_code=%s elapsed_ms=%.2f"
            ),
            operation,
            fallback_decision.primary_model,
            fallback_route.model,
            fallback_route.tier,
            fallback_decision.primary_error_code,
            elapsed_ms,
        )

    def _log_fallback_failed(
        self,
        *,
        operation: str,
        fallback_decision: LLMFallbackDecision,
        fallback_exception: AppException,
        elapsed_ms: float,
    ) -> None:
        fallback_route = fallback_decision.fallback_route
        if fallback_route is None:
            return
        logger.warning(
            (
                "llm_fallback_failed operation=%s primary_model=%s "
                "fallback_model=%s fallback_tier=%s primary_code=%s "
                "fallback_code=%s elapsed_ms=%.2f"
            ),
            operation,
            fallback_decision.primary_model,
            fallback_route.model,
            fallback_route.tier,
            fallback_decision.primary_error_code,
            fallback_exception.code,
            elapsed_ms,
        )

    def _log_cost_control_decision(
        self,
        *,
        operation: str,
        route_decision: LLMModelRouteDecision,
        cost_decision: LLMCostControlDecision,
    ) -> None:
        logger.info(
            (
                "llm_cost_control_decision operation=%s model=%s action=%s "
                "reason=%s estimated_input_tokens=%s requested_max_output_tokens=%s "
                "effective_max_output_tokens=%s reserved_total_tokens=%s "
                "fallback_allowed=%s estimated_cost=%s currency=%s"
            ),
            operation,
            route_decision.model,
            cost_decision.action,
            cost_decision.reason,
            cost_decision.estimated_input_tokens,
            cost_decision.requested_max_output_tokens,
            cost_decision.effective_max_output_tokens,
            cost_decision.reserved_total_tokens,
            cost_decision.fallback_allowed,
            cost_decision.estimated_cost.total_cost,
            cost_decision.estimated_cost.currency,
        )

    def _log_retry_decision(
        self,
        *,
        operation: str,
        route_decision: LLMModelRouteDecision,
        retry_decision: LLMRetryDecision,
    ) -> None:
        logger.info(
            (
                "llm_retry_decision operation=%s model=%s error_code=%s "
                "attempt=%s max_attempts=%s should_retry=%s reason=%s "
                "next_attempt=%s next_delay=%s"
            ),
            operation,
            route_decision.model,
            retry_decision.error_code,
            retry_decision.attempt_number,
            retry_decision.max_attempts,
            retry_decision.should_retry,
            retry_decision.reason,
            retry_decision.next_attempt_number,
            retry_decision.next_delay_seconds,
        )

    def _log_timeout_budget_decision(
        self,
        *,
        operation: str,
        route_decision: LLMModelRouteDecision,
        timeout_decision: LLMTimeoutBudgetDecision,
    ) -> None:
        logger.info(
            (
                "llm_timeout_budget_decision operation=%s model=%s phase=%s "
                "allowed=%s reason=%s elapsed_seconds=%.3f "
                "total_timeout_seconds=%.3f remaining_seconds=%.3f "
                "required_seconds=%.3f next_delay_seconds=%.3f"
            ),
            operation,
            route_decision.model,
            timeout_decision.phase,
            timeout_decision.allowed,
            timeout_decision.reason,
            timeout_decision.elapsed_seconds,
            timeout_decision.total_timeout_seconds,
            timeout_decision.remaining_seconds,
            timeout_decision.required_seconds,
            timeout_decision.next_delay_seconds,
        )

    def _to_app_exception(self, exc: Exception) -> AppException:
        if isinstance(exc, AppException):
            return exc
        return map_openai_error_to_app_exception(exc)

    def _build_cost_control_decision(
        self,
        serialized_messages: list[dict[str, str]],
    ) -> LLMCostControlDecision:
        return build_llm_cost_control_decision(
            self.settings,
            serialized_messages=serialized_messages,
            requested_max_output_tokens=self.settings.max_output_tokens,
            pricing=self._build_token_pricing(),
        )

    def _raise_cost_control_exception(
        self,
        cost_decision: LLMCostControlDecision,
    ) -> None:
        if cost_decision.reason == "input_tokens_exceeded":
            message = "本次问题或上下文过长，已超过当前 AI 服务的输入预算。"
        elif cost_decision.reason == "estimated_cost_exceeded":
            message = "本次请求预计成本过高，已超过当前 AI 服务的单次请求预算。"
        else:
            message = "本次请求预计消耗过高，已超过当前 AI 服务的单次请求预算。"
        raise AppException(
            code="LLM_COST_BUDGET_EXCEEDED",
            message=message,
            status_code=429,
        )

    def _raise_timeout_budget_exception(
        self,
        timeout_decision: LLMTimeoutBudgetDecision,
    ) -> None:
        raise AppException(
            code="LLM_TOTAL_TIMEOUT_EXCEEDED",
            message="本次模型调用预计会超过总超时预算，请稍后重试。",
            status_code=504,
        )

    def _apply_cost_control_to_fallback(
        self,
        fallback_decision: LLMFallbackDecision,
        cost_decision: LLMCostControlDecision,
    ) -> LLMFallbackDecision:
        if fallback_decision.should_attempt and not cost_decision.fallback_allowed:
            return disable_fallback_by_cost_control(fallback_decision)
        return fallback_decision

    def _create_chat_completion_with_retry(
        self,
        route_decision: LLMModelRouteDecision,
        serialized_messages: list[dict[str, str]],
        *,
        max_tokens: int,
        started_at: float,
    ) -> Any:
        return self._create_completion_with_retry(
            operation="chat",
            route_decision=route_decision,
            serialized_messages=serialized_messages,
            max_tokens=max_tokens,
            stream=False,
            started_at=started_at,
        )

    def _create_stream_completion_with_retry(
        self,
        route_decision: LLMModelRouteDecision,
        serialized_messages: list[dict[str, str]],
        *,
        max_tokens: int,
        started_at: float,
    ) -> Any:
        return self._create_completion_with_retry(
            operation="stream_chat",
            route_decision=route_decision,
            serialized_messages=serialized_messages,
            max_tokens=max_tokens,
            stream=True,
            started_at=started_at,
        )

    def _create_completion_with_retry(
        self,
        *,
        operation: str,
        route_decision: LLMModelRouteDecision,
        serialized_messages: list[dict[str, str]],
        max_tokens: int,
        stream: bool,
        started_at: float,
    ) -> Any:
        attempt_number = 1
        while True:
            try:
                if stream:
                    return self._create_stream_completion(
                        route_decision,
                        serialized_messages,
                        max_tokens=max_tokens,
                    )
                return self._create_chat_completion(
                    route_decision,
                    serialized_messages,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                app_exception = self._to_app_exception(exc)
                retry_decision = build_llm_retry_decision(
                    self.settings,
                    error_code=app_exception.code,
                    attempt_number=attempt_number,
                    retry_after_seconds=extract_retry_after_seconds(exc),
                )
                self._log_retry_decision(
                    operation=operation,
                    route_decision=route_decision,
                    retry_decision=retry_decision,
                )
                if not retry_decision.should_retry:
                    raise app_exception from exc

                timeout_decision = build_llm_timeout_budget_decision(
                    self.settings,
                    phase="retry",
                    elapsed_seconds=self._elapsed_seconds(started_at),
                    next_delay_seconds=retry_decision.next_delay_seconds or 0.0,
                )
                self._log_timeout_budget_decision(
                    operation=operation,
                    route_decision=route_decision,
                    timeout_decision=timeout_decision,
                )
                if not timeout_decision.allowed:
                    self._raise_timeout_budget_exception(timeout_decision)

                if retry_decision.next_delay_seconds is not None:
                    self._sleep_func(retry_decision.next_delay_seconds)
                attempt_number = retry_decision.next_attempt_number or (
                    attempt_number + 1
                )

    def _create_chat_completion(
        self,
        route_decision: LLMModelRouteDecision,
        serialized_messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> Any:
        return self._get_client().chat.completions.create(
            model=route_decision.model,
            messages=serialized_messages,
            max_tokens=max_tokens,
        )

    def _create_stream_completion(
        self,
        route_decision: LLMModelRouteDecision,
        serialized_messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> Any:
        return self._get_client().chat.completions.create(
            model=route_decision.model,
            messages=serialized_messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
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

        route_decision = route_llm_model(
            self.settings,
            operation="chat",
            input_text=user_message,
        )
        messages = build_chat_messages(user_message, history=history)
        serialized_messages = serialize_chat_messages(messages)
        cost_decision = self._build_cost_control_decision(serialized_messages)
        self._log_cost_control_decision(
            operation="chat",
            route_decision=route_decision,
            cost_decision=cost_decision,
        )
        if cost_decision.should_block:
            self._raise_cost_control_exception(cost_decision)

        start_time = self._now()
        try:
            completion = self._create_chat_completion_with_retry(
                route_decision,
                serialized_messages,
                max_tokens=cost_decision.effective_max_output_tokens,
                started_at=start_time,
            )
            reply = extract_first_reply(completion)
        except Exception as exc:
            app_exception = self._to_app_exception(exc)
            fallback_decision = build_llm_fallback_decision(
                self.settings,
                primary_route=route_decision,
                error_code=app_exception.code,
            )
            fallback_decision = self._apply_cost_control_to_fallback(
                fallback_decision,
                cost_decision,
            )
            self._log_failure(
                app_exception,
                self._elapsed_ms(start_time),
                route_decision=route_decision,
                fallback_decision=fallback_decision,
                exc_info=not isinstance(exc, AppException),
            )
            if (
                not fallback_decision.should_attempt
                or fallback_decision.fallback_route is None
            ):
                raise app_exception from exc

            fallback_route = fallback_decision.fallback_route
            timeout_decision = build_llm_timeout_budget_decision(
                self.settings,
                phase="fallback",
                elapsed_seconds=self._elapsed_seconds(start_time),
            )
            self._log_timeout_budget_decision(
                operation="chat",
                route_decision=fallback_route,
                timeout_decision=timeout_decision,
            )
            if not timeout_decision.allowed:
                self._raise_timeout_budget_exception(timeout_decision)

            self._log_fallback_started(
                operation="chat",
                fallback_decision=fallback_decision,
            )
            fallback_start_time = self._now()
            try:
                completion = self._create_chat_completion(
                    fallback_route,
                    serialized_messages,
                    max_tokens=cost_decision.effective_max_output_tokens,
                )
                reply = extract_first_reply(completion)
            except Exception as fallback_exc:
                fallback_app_exception = self._to_app_exception(fallback_exc)
                fallback_elapsed_ms = self._elapsed_ms(fallback_start_time)
                self._log_failure(
                    fallback_app_exception,
                    fallback_elapsed_ms,
                    route_decision=fallback_route,
                    exc_info=not isinstance(fallback_exc, AppException),
                )
                self._log_fallback_failed(
                    operation="chat",
                    fallback_decision=fallback_decision,
                    fallback_exception=fallback_app_exception,
                    elapsed_ms=fallback_elapsed_ms,
                )
                raise fallback_app_exception from fallback_exc

            fallback_elapsed_ms = self._elapsed_ms(fallback_start_time)
            self._log_success(
                fallback_elapsed_ms,
                extract_token_usage(completion),
                fallback_route,
            )
            self._log_fallback_succeeded(
                operation="chat",
                fallback_decision=fallback_decision,
                elapsed_ms=fallback_elapsed_ms,
            )
            return reply

        self._log_success(
            self._elapsed_ms(start_time),
            extract_token_usage(completion),
            route_decision,
        )
        return reply

    def _iter_stream_reply_chunks(
        self,
        stream: Iterator[Any],
        start_time: float,
        route_decision: LLMModelRouteDecision,
    ) -> Iterator[str]:
        usage = LLMTokenUsage()
        chunk_count = 0
        content_chunk_count = 0

        try:
            for chunk in stream:
                chunk_count += 1

                chunk_usage = extract_token_usage(chunk)
                if has_token_usage(chunk_usage):
                    usage = chunk_usage

                content = extract_stream_delta_content(chunk)
                if content is None:
                    continue

                content_chunk_count += 1
                yield content
        except AppException as exc:
            self._log_stream_failure(
                exc,
                self._elapsed_ms(start_time),
                chunk_count=chunk_count,
                content_chunk_count=content_chunk_count,
                route_decision=route_decision,
            )
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_stream_failure(
                app_exception,
                self._elapsed_ms(start_time),
                chunk_count=chunk_count,
                content_chunk_count=content_chunk_count,
                route_decision=route_decision,
                exc_info=True,
            )
            raise app_exception from exc

        self._log_stream_success(
            self._elapsed_ms(start_time),
            usage,
            chunk_count=chunk_count,
            content_chunk_count=content_chunk_count,
            route_decision=route_decision,
        )

    def stream_reply(
        self,
        user_message: str,
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> Iterator[str]:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        route_decision = route_llm_model(
            self.settings,
            operation="stream_chat",
            input_text=user_message,
        )
        messages = build_chat_messages(user_message, history=history)
        serialized_messages = serialize_chat_messages(messages)
        cost_decision = self._build_cost_control_decision(serialized_messages)
        self._log_cost_control_decision(
            operation="stream_chat",
            route_decision=route_decision,
            cost_decision=cost_decision,
        )
        if cost_decision.should_block:
            self._raise_cost_control_exception(cost_decision)

        start_time = self._now()
        try:
            stream = self._create_stream_completion_with_retry(
                route_decision,
                serialized_messages,
                max_tokens=cost_decision.effective_max_output_tokens,
                started_at=start_time,
            )
        except Exception as exc:
            app_exception = self._to_app_exception(exc)
            fallback_decision = build_llm_fallback_decision(
                self.settings,
                primary_route=route_decision,
                error_code=app_exception.code,
            )
            fallback_decision = self._apply_cost_control_to_fallback(
                fallback_decision,
                cost_decision,
            )
            self._log_stream_failure(
                app_exception,
                self._elapsed_ms(start_time),
                chunk_count=0,
                content_chunk_count=0,
                route_decision=route_decision,
                fallback_decision=fallback_decision,
                exc_info=not isinstance(exc, AppException),
            )
            if (
                not fallback_decision.should_attempt
                or fallback_decision.fallback_route is None
            ):
                raise app_exception from exc

            fallback_route = fallback_decision.fallback_route
            timeout_decision = build_llm_timeout_budget_decision(
                self.settings,
                phase="fallback",
                elapsed_seconds=self._elapsed_seconds(start_time),
            )
            self._log_timeout_budget_decision(
                operation="stream_chat",
                route_decision=fallback_route,
                timeout_decision=timeout_decision,
            )
            if not timeout_decision.allowed:
                self._raise_timeout_budget_exception(timeout_decision)

            self._log_fallback_started(
                operation="stream_chat",
                fallback_decision=fallback_decision,
            )
            fallback_start_time = self._now()
            try:
                stream = self._create_stream_completion(
                    fallback_route,
                    serialized_messages,
                    max_tokens=cost_decision.effective_max_output_tokens,
                )
            except Exception as fallback_exc:
                fallback_app_exception = self._to_app_exception(fallback_exc)
                fallback_elapsed_ms = self._elapsed_ms(fallback_start_time)
                self._log_stream_failure(
                    fallback_app_exception,
                    fallback_elapsed_ms,
                    chunk_count=0,
                    content_chunk_count=0,
                    route_decision=fallback_route,
                    exc_info=not isinstance(fallback_exc, AppException),
                )
                self._log_fallback_failed(
                    operation="stream_chat",
                    fallback_decision=fallback_decision,
                    fallback_exception=fallback_app_exception,
                    elapsed_ms=fallback_elapsed_ms,
                )
                raise fallback_app_exception from fallback_exc

            self._log_fallback_succeeded(
                operation="stream_chat",
                fallback_decision=fallback_decision,
                elapsed_ms=self._elapsed_ms(fallback_start_time),
            )
            return self._iter_stream_reply_chunks(
                iter(stream),
                fallback_start_time,
                fallback_route,
            )

        return self._iter_stream_reply_chunks(iter(stream), start_time, route_decision)


def create_llm_chat_service(settings: Settings) -> LLMChatService:
    return LLMChatService(settings)
