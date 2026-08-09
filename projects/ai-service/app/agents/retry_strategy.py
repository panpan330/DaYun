from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import httpx

from app.core.config import Settings


RetryDependencyKind = Literal[
    "llm",
    "embedding",
    "java_read_tool",
    "java_write_tool",
    "vector_store",
    "milvus",
    "rag_generation",
]
RetryFailureCategory = Literal[
    "connection_error",
    "timeout",
    "rate_limited",
    "server_error",
    "conflict",
    "client_error",
    "validation_error",
    "auth_error",
    "permission_error",
    "not_found",
    "business_rule",
    "unknown",
]
RetryDecisionReason = Literal[
    "retry_allowed",
    "retry_after_allowed",
    "retry_disabled",
    "max_retries_exhausted",
    "failure_not_retryable",
    "status_not_retryable",
    "idempotency_key_required",
]

DEFAULT_RETRY_INITIAL_DELAY_SECONDS = 0.25
DEFAULT_RETRY_MULTIPLIER = 2.0
DEFAULT_RETRY_MAX_DELAY_SECONDS = 2.0
DEFAULT_RETRY_JITTER_SECONDS = 0.1

RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
CLIENT_ERROR_STATUS_CODES = frozenset({400, 422})
AUTH_ERROR_STATUS_CODES = frozenset({401})
PERMISSION_ERROR_STATUS_CODES = frozenset({403})
NOT_FOUND_STATUS_CODES = frozenset({404})

RETRY_HIGH_CARDINALITY_ATTRIBUTE_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "thread_id",
        "session_id",
        "actor_id",
        "user_id",
        "conversation_id",
        "request_id",
        "idempotency_key",
        "user_message",
        "prompt",
        "messages",
        "final_answer",
        "raw_response",
        "tool_args",
    }
)


@dataclass(frozen=True)
class RetryBackoff:
    initial_delay_seconds: float = DEFAULT_RETRY_INITIAL_DELAY_SECONDS
    multiplier: float = DEFAULT_RETRY_MULTIPLIER
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS
    jitter_seconds: float = DEFAULT_RETRY_JITTER_SECONDS

    def __post_init__(self) -> None:
        _validate_positive_number(
            self.initial_delay_seconds,
            "initial_delay_seconds",
        )
        _validate_positive_number(self.multiplier, "multiplier")
        _validate_positive_number(self.max_delay_seconds, "max_delay_seconds")
        _validate_non_negative_number(self.jitter_seconds, "jitter_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be greater than or equal to 1.")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be greater than or equal to "
                "initial_delay_seconds."
            )

    def delay_for_retry(
        self,
        retry_number: int,
        *,
        jitter_ratio: float | None = None,
    ) -> float:
        if retry_number < 1:
            raise ValueError("retry_number must be greater than or equal to 1.")
        if jitter_ratio is not None:
            _validate_jitter_ratio(jitter_ratio)

        base_delay = self.initial_delay_seconds * (
            self.multiplier ** (retry_number - 1)
        )
        capped_delay = min(base_delay, self.max_delay_seconds)
        if jitter_ratio is None or self.jitter_seconds == 0:
            return _round_seconds(capped_delay)

        jitter = self.jitter_seconds * jitter_ratio
        return _round_seconds(min(capped_delay + jitter, self.max_delay_seconds))

    def build_delay_schedule(
        self,
        max_retries: int,
        *,
        jitter_ratio: float | None = None,
    ) -> list[float]:
        if max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")
        return [
            self.delay_for_retry(retry_number, jitter_ratio=jitter_ratio)
            for retry_number in range(1, max_retries + 1)
        ]


@dataclass(frozen=True)
class TicketAgentRetryPolicy:
    dependency_kind: RetryDependencyKind
    operation: str
    max_retries: int
    backoff: RetryBackoff
    retryable_categories: frozenset[RetryFailureCategory]
    retryable_status_codes: frozenset[int]
    fallback_allowed: bool
    user_message: str
    requires_idempotency_key: bool = False
    retry_all_server_errors: bool = True
    respect_retry_after_header: bool = True
    cost_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("operation must not be blank.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")
        if self.max_retries > 0 and not self.retryable_categories:
            raise ValueError("retryable_categories must not be empty.")
        for status_code in self.retryable_status_codes:
            _validate_http_status_code(status_code)

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def delay_schedule(self, *, jitter_ratio: float | None = None) -> list[float]:
        return self.backoff.build_delay_schedule(
            self.max_retries,
            jitter_ratio=jitter_ratio,
        )


@dataclass(frozen=True)
class RetryDecision:
    dependency_kind: RetryDependencyKind
    operation: str
    attempt_number: int
    max_attempts: int
    failure_category: RetryFailureCategory
    should_retry: bool
    reason: RetryDecisionReason
    fallback_allowed: bool
    status_code: int | None = None
    next_delay_seconds: float | None = None
    next_attempt_number: int | None = None
    blocked_by_idempotency: bool = False
    cost_sensitive: bool = False

    def log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "failure_category": self.failure_category,
            "should_retry": self.should_retry,
            "retry_decision_reason": self.reason,
            "fallback_allowed": self.fallback_allowed,
            "blocked_by_idempotency": self.blocked_by_idempotency,
            "cost_sensitive": self.cost_sensitive,
        }
        if self.status_code is not None:
            fields["status_code"] = self.status_code
        if self.next_delay_seconds is not None:
            fields["next_delay_seconds"] = self.next_delay_seconds
        if self.next_attempt_number is not None:
            fields["next_attempt_number"] = self.next_attempt_number
        return fields

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "failure_category": self.failure_category,
            "should_retry": self.should_retry,
            "retry_decision_reason": self.reason,
            "fallback_allowed": self.fallback_allowed,
            "blocked_by_idempotency": self.blocked_by_idempotency,
            "cost_sensitive": self.cost_sensitive,
        }
        if self.status_code is not None:
            fields["status_code"] = self.status_code
        return fields


def build_default_retry_backoff() -> RetryBackoff:
    return RetryBackoff()


def build_ticket_agent_retry_policies(
    settings: Settings,
) -> dict[str, TicketAgentRetryPolicy]:
    default_backoff = build_default_retry_backoff()
    return {
        "llm.intent_classification": TicketAgentRetryPolicy(
            dependency_kind="llm",
            operation="ticket_intent_classification",
            max_retries=settings.llm_max_retries,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {
                    "connection_error",
                    "timeout",
                    "rate_limited",
                    "server_error",
                    "conflict",
                }
            ),
            retryable_status_codes=RETRYABLE_HTTP_STATUS_CODES,
            fallback_allowed=True,
            user_message="模型意图识别临时失败，系统会短暂重试后再决定是否兜底。",
            cost_sensitive=True,
        ),
        "llm.field_extraction": TicketAgentRetryPolicy(
            dependency_kind="llm",
            operation="ticket_field_extraction",
            max_retries=settings.llm_max_retries,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {
                    "connection_error",
                    "timeout",
                    "rate_limited",
                    "server_error",
                    "conflict",
                }
            ),
            retryable_status_codes=RETRYABLE_HTTP_STATUS_CODES,
            fallback_allowed=True,
            user_message="模型字段提取临时失败，系统会短暂重试后再决定是否兜底。",
            cost_sensitive=True,
        ),
        "embedding.create": TicketAgentRetryPolicy(
            dependency_kind="embedding",
            operation="create_embeddings",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=False,
            user_message="向量生成临时失败，系统会短暂重试一次。",
            cost_sensitive=True,
        ),
        "java.query_order": TicketAgentRetryPolicy(
            dependency_kind="java_read_tool",
            operation="query_order",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=True,
            user_message="订单查询工具临时失败，系统会短暂重试一次。",
        ),
        "java.create_ticket": TicketAgentRetryPolicy(
            dependency_kind="java_write_tool",
            operation="create_ticket",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=False,
            requires_idempotency_key=True,
            user_message="创建工单工具临时失败，只有具备幂等键时才允许自动重试。",
        ),
        "qdrant.vector_search": TicketAgentRetryPolicy(
            dependency_kind="vector_store",
            operation="vector_search",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=True,
            user_message="Qdrant 检索临时失败，系统会短暂重试一次，仍失败则走无上下文兜底。",
        ),
        "milvus.vector_search": TicketAgentRetryPolicy(
            dependency_kind="milvus",
            operation="vector_search",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=True,
            user_message="Milvus 检索临时失败，系统会短暂重试一次，仍失败则走无上下文兜底。",
        ),
        "rag.generate_answer": TicketAgentRetryPolicy(
            dependency_kind="rag_generation",
            operation="generate_answer",
            max_retries=1,
            backoff=default_backoff,
            retryable_categories=frozenset(
                {"connection_error", "timeout", "rate_limited", "server_error"}
            ),
            retryable_status_codes=frozenset({408, 429, 500, 502, 503, 504}),
            fallback_allowed=True,
            user_message="RAG 回答生成临时失败，系统会短暂重试一次，仍失败则返回安全兜底。",
            cost_sensitive=True,
        ),
    }


def classify_http_status_for_retry(status_code: int) -> RetryFailureCategory:
    _validate_http_status_code(status_code)
    if status_code == 408:
        return "timeout"
    if status_code == 409:
        return "conflict"
    if status_code == 429:
        return "rate_limited"
    if 500 <= status_code <= 599:
        return "server_error"
    if status_code in CLIENT_ERROR_STATUS_CODES:
        return "validation_error"
    if status_code in AUTH_ERROR_STATUS_CODES:
        return "auth_error"
    if status_code in PERMISSION_ERROR_STATUS_CODES:
        return "permission_error"
    if status_code in NOT_FOUND_STATUS_CODES:
        return "not_found"
    if 400 <= status_code <= 499:
        return "client_error"
    return "unknown"


def classify_error_code_for_retry(error_code: str) -> RetryFailureCategory:
    normalized = error_code.strip().upper()
    if not normalized:
        return "unknown"
    if "TIMEOUT" in normalized:
        return "timeout"
    if "RATE_LIMIT" in normalized or "TOO_MANY" in normalized:
        return "rate_limited"
    if "CONNECTION" in normalized or "NETWORK" in normalized:
        return "connection_error"
    if "AUTH" in normalized or "API_KEY" in normalized or "UNAUTHORIZED" in normalized:
        return "auth_error"
    if "PERMISSION" in normalized or "FORBIDDEN" in normalized:
        return "permission_error"
    if "NOT_FOUND" in normalized:
        return "not_found"
    if "VALIDATION" in normalized or "INVALID" in normalized or "BAD_REQUEST" in normalized:
        return "validation_error"
    if "CONFLICT" in normalized:
        return "conflict"
    if "BUSINESS" in normalized:
        return "business_rule"
    if "SERVER" in normalized or "INTERNAL" in normalized:
        return "server_error"
    return "unknown"


def classify_exception_for_retry(exc: BaseException) -> RetryFailureCategory:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connection_error"
    if isinstance(exc, httpx.NetworkError):
        return "connection_error"
    if isinstance(exc, TimeoutError):
        return "timeout"

    exc_type_name = exc.__class__.__name__.casefold()
    if "ratelimit" in exc_type_name or "rate_limit" in exc_type_name:
        return "rate_limited"
    if "apitimeouterror" in exc_type_name or "timeout" in exc_type_name:
        return "timeout"
    if "apiconnectionerror" in exc_type_name or "connection" in exc_type_name:
        return "connection_error"
    if "internalservererror" in exc_type_name or "server" in exc_type_name:
        return "server_error"
    if "authentication" in exc_type_name or "api_key" in exc_type_name:
        return "auth_error"
    if "permission" in exc_type_name:
        return "permission_error"
    if "notfound" in exc_type_name:
        return "not_found"
    if "badrequest" in exc_type_name or "validation" in exc_type_name:
        return "validation_error"
    if "conflict" in exc_type_name:
        return "conflict"
    return "unknown"


def classify_retry_failure(
    *,
    exc: BaseException | None = None,
    status_code: int | None = None,
    error_code: str | None = None,
) -> RetryFailureCategory:
    if status_code is not None:
        return classify_http_status_for_retry(status_code)
    if error_code is not None:
        return classify_error_code_for_retry(error_code)
    if exc is not None:
        return classify_exception_for_retry(exc)
    return "unknown"


def decide_retry(
    policy: TicketAgentRetryPolicy,
    *,
    attempt_number: int,
    failure_category: RetryFailureCategory,
    status_code: int | None = None,
    idempotency_key_present: bool = False,
    retry_after_seconds: float | None = None,
    jitter_ratio: float | None = None,
) -> RetryDecision:
    if attempt_number < 1:
        raise ValueError("attempt_number must be greater than or equal to 1.")
    if status_code is not None:
        _validate_http_status_code(status_code)
    normalized_retry_after = _normalize_retry_after_seconds(retry_after_seconds)

    if policy.max_retries == 0:
        return _blocked_decision(
            policy,
            attempt_number=attempt_number,
            failure_category=failure_category,
            status_code=status_code,
            reason="retry_disabled",
        )
    if failure_category not in policy.retryable_categories:
        return _blocked_decision(
            policy,
            attempt_number=attempt_number,
            failure_category=failure_category,
            status_code=status_code,
            reason="failure_not_retryable",
        )
    if status_code is not None and not _is_status_code_retryable(policy, status_code):
        return _blocked_decision(
            policy,
            attempt_number=attempt_number,
            failure_category=failure_category,
            status_code=status_code,
            reason="status_not_retryable",
        )
    if policy.requires_idempotency_key and not idempotency_key_present:
        return _blocked_decision(
            policy,
            attempt_number=attempt_number,
            failure_category=failure_category,
            status_code=status_code,
            reason="idempotency_key_required",
            blocked_by_idempotency=True,
        )
    if attempt_number >= policy.max_attempts:
        return _blocked_decision(
            policy,
            attempt_number=attempt_number,
            failure_category=failure_category,
            status_code=status_code,
            reason="max_retries_exhausted",
        )

    retry_number = attempt_number
    if (
        policy.respect_retry_after_header
        and normalized_retry_after is not None
        and failure_category == "rate_limited"
    ):
        next_delay_seconds = min(
            normalized_retry_after,
            policy.backoff.max_delay_seconds,
        )
        reason: RetryDecisionReason = "retry_after_allowed"
    else:
        next_delay_seconds = policy.backoff.delay_for_retry(
            retry_number,
            jitter_ratio=jitter_ratio,
        )
        reason = "retry_allowed"

    return RetryDecision(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        attempt_number=attempt_number,
        max_attempts=policy.max_attempts,
        failure_category=failure_category,
        should_retry=True,
        reason=reason,
        fallback_allowed=policy.fallback_allowed,
        status_code=status_code,
        next_delay_seconds=next_delay_seconds,
        next_attempt_number=attempt_number + 1,
        cost_sensitive=policy.cost_sensitive,
    )


def sanitize_retry_metric_attributes(
    attributes: dict[str, object],
) -> dict[str, str | int | float | bool]:
    safe_attributes: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in RETRY_HIGH_CARDINALITY_ATTRIBUTE_KEYS:
            continue
        normalized_value = _normalize_attribute_value(value)
        if normalized_value is not None:
            safe_attributes[str(key)] = normalized_value
    return safe_attributes


def _blocked_decision(
    policy: TicketAgentRetryPolicy,
    *,
    attempt_number: int,
    failure_category: RetryFailureCategory,
    status_code: int | None,
    reason: RetryDecisionReason,
    blocked_by_idempotency: bool = False,
) -> RetryDecision:
    return RetryDecision(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        attempt_number=attempt_number,
        max_attempts=policy.max_attempts,
        failure_category=failure_category,
        should_retry=False,
        reason=reason,
        fallback_allowed=policy.fallback_allowed,
        status_code=status_code,
        blocked_by_idempotency=blocked_by_idempotency,
        cost_sensitive=policy.cost_sensitive,
    )


def _is_status_code_retryable(
    policy: TicketAgentRetryPolicy,
    status_code: int,
) -> bool:
    if status_code in policy.retryable_status_codes:
        return True
    return (
        policy.retry_all_server_errors
        and 500 <= status_code <= 599
        and "server_error" in policy.retryable_categories
    )


def _validate_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")


def _validate_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")


def _validate_jitter_ratio(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("jitter_ratio must be a number.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("jitter_ratio must be between 0 and 1.")


def _validate_http_status_code(status_code: int) -> None:
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise ValueError("status_code must be an integer.")
    if not 100 <= status_code <= 599:
        raise ValueError("status_code must be between 100 and 599.")


def _normalize_retry_after_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    _validate_positive_number(value, "retry_after_seconds")
    return _round_seconds(value)


def _round_seconds(value: float) -> float:
    return max(0.001, round(value, 3))


def _normalize_attribute_value(value: object) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 6)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return normalized
    return None
