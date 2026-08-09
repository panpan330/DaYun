from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import httpx

from app.core.config import Settings


TimeoutPhase = Literal[
    "connect",
    "read",
    "write",
    "pool",
    "total",
    "operation",
    "unknown",
]
TimeoutDependencyKind = Literal[
    "llm",
    "embedding",
    "java_read_tool",
    "java_write_tool",
    "vector_store",
    "milvus",
    "rag_generation",
]
TimeoutRecoveryAction = Literal[
    "retry_later",
    "return_safe_fallback",
    "use_cache_or_return_no_context",
    "require_idempotency_before_retry",
    "investigate_system",
]

DEFAULT_CONNECT_TIMEOUT_RATIO = 0.2
DEFAULT_WRITE_TIMEOUT_RATIO = 0.4
DEFAULT_POOL_TIMEOUT_RATIO = 0.1

TIMEOUT_HIGH_CARDINALITY_ATTRIBUTE_KEYS = frozenset(
    {
        "trace_id",
        "span_id",
        "thread_id",
        "session_id",
        "actor_id",
        "user_id",
        "conversation_id",
        "user_message",
        "prompt",
        "messages",
        "final_answer",
        "raw_response",
    }
)


@dataclass(frozen=True)
class TimeoutBudget:
    total_seconds: float
    connect_seconds: float | None = None
    read_seconds: float | None = None
    write_seconds: float | None = None
    pool_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_positive_timeout(self.total_seconds, "total_seconds")
        for field_name, value in (
            ("connect_seconds", self.connect_seconds),
            ("read_seconds", self.read_seconds),
            ("write_seconds", self.write_seconds),
            ("pool_seconds", self.pool_seconds),
        ):
            if value is not None:
                _validate_positive_timeout(value, field_name)

    def resolved_connect_seconds(self) -> float:
        return self.connect_seconds or _round_timeout(
            self.total_seconds * DEFAULT_CONNECT_TIMEOUT_RATIO
        )

    def resolved_read_seconds(self) -> float:
        return self.read_seconds or self.total_seconds

    def resolved_write_seconds(self) -> float:
        return self.write_seconds or _round_timeout(
            min(self.total_seconds, self.total_seconds * DEFAULT_WRITE_TIMEOUT_RATIO)
        )

    def resolved_pool_seconds(self) -> float:
        return self.pool_seconds or _round_timeout(
            self.total_seconds * DEFAULT_POOL_TIMEOUT_RATIO
        )

    def as_httpx_timeout_kwargs(self) -> dict[str, float]:
        return {
            "timeout": self.total_seconds,
            "connect": self.resolved_connect_seconds(),
            "read": self.resolved_read_seconds(),
            "write": self.resolved_write_seconds(),
            "pool": self.resolved_pool_seconds(),
        }


@dataclass(frozen=True)
class TicketAgentTimeoutPolicy:
    dependency_kind: TimeoutDependencyKind
    operation: str
    budget: TimeoutBudget
    error_code: str
    status_code: int
    retryable: bool
    max_retries: int
    fallback_allowed: bool
    recovery_action: TimeoutRecoveryAction
    requires_idempotency_key: bool = False
    user_message: str = "外部依赖响应超时，请稍后重试。"

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("operation must not be blank.")
        if not self.error_code.strip():
            raise ValueError("error_code must not be blank.")
        if self.status_code < 400:
            raise ValueError("status_code must be an error status.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")


@dataclass(frozen=True)
class TimeoutFailure:
    dependency_kind: TimeoutDependencyKind
    operation: str
    phase: TimeoutPhase
    error_code: str
    status_code: int
    retryable: bool
    fallback_allowed: bool
    recovery_action: TimeoutRecoveryAction
    user_message: str
    elapsed_ms: float | None = None

    def log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "timeout_phase": self.phase,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "fallback_allowed": self.fallback_allowed,
            "recovery_action": self.recovery_action,
        }
        if self.elapsed_ms is not None:
            fields["elapsed_ms"] = round(self.elapsed_ms, 2)
        return fields

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        return {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "timeout_phase": self.phase,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "fallback_allowed": self.fallback_allowed,
        }


def build_timeout_budget(total_seconds: float) -> TimeoutBudget:
    return TimeoutBudget(total_seconds=total_seconds)


def build_ticket_agent_timeout_policies(
    settings: Settings,
) -> dict[str, TicketAgentTimeoutPolicy]:
    return {
        "llm.intent_classification": TicketAgentTimeoutPolicy(
            dependency_kind="llm",
            operation="ticket_intent_classification",
            budget=build_timeout_budget(settings.request_timeout_seconds),
            error_code="LLM_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=settings.llm_max_retries,
            fallback_allowed=True,
            recovery_action="return_safe_fallback",
            user_message="模型意图识别响应超时，系统会使用安全兜底策略。",
        ),
        "llm.field_extraction": TicketAgentTimeoutPolicy(
            dependency_kind="llm",
            operation="ticket_field_extraction",
            budget=build_timeout_budget(settings.request_timeout_seconds),
            error_code="LLM_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=settings.llm_max_retries,
            fallback_allowed=True,
            recovery_action="return_safe_fallback",
            user_message="模型字段提取响应超时，系统会使用安全兜底策略。",
        ),
        "embedding.create": TicketAgentTimeoutPolicy(
            dependency_kind="embedding",
            operation="create_embeddings",
            budget=build_timeout_budget(settings.request_timeout_seconds),
            error_code="EMBEDDING_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=False,
            recovery_action="retry_later",
            user_message="向量生成响应超时，请稍后重试。",
        ),
        "java.query_order": TicketAgentTimeoutPolicy(
            dependency_kind="java_read_tool",
            operation="query_order",
            budget=build_timeout_budget(
                settings.resolved_java_business_service_timeout_seconds
            ),
            error_code="TOOL_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="retry_later",
            user_message="订单查询工具调用超时，请稍后重试。",
        ),
        "java.create_ticket": TicketAgentTimeoutPolicy(
            dependency_kind="java_write_tool",
            operation="create_ticket",
            budget=build_timeout_budget(
                settings.resolved_java_business_service_timeout_seconds
            ),
            error_code="TOOL_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=False,
            recovery_action="require_idempotency_before_retry",
            requires_idempotency_key=True,
            user_message="创建工单工具调用超时，请稍后查看是否已创建成功。",
        ),
        "qdrant.vector_search": TicketAgentTimeoutPolicy(
            dependency_kind="vector_store",
            operation="vector_search",
            budget=build_timeout_budget(settings.qdrant_timeout_seconds),
            error_code="RAG_VECTOR_STORE_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="use_cache_or_return_no_context",
            user_message="知识库检索响应超时，系统会尝试使用缓存或返回安全兜底答案。",
        ),
        "milvus.vector_search": TicketAgentTimeoutPolicy(
            dependency_kind="milvus",
            operation="vector_search",
            budget=build_timeout_budget(settings.milvus_timeout_seconds),
            error_code="RAG_VECTOR_STORE_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="use_cache_or_return_no_context",
            user_message="Milvus 检索响应超时，系统会尝试使用缓存或返回安全兜底答案。",
        ),
        "rag.generate_answer": TicketAgentTimeoutPolicy(
            dependency_kind="rag_generation",
            operation="generate_answer",
            budget=build_timeout_budget(settings.request_timeout_seconds),
            error_code="RAG_GENERATION_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="return_safe_fallback",
            user_message="知识库回答生成超时，系统会返回安全兜底答案。",
        ),
    }


def classify_timeout_phase(exc: BaseException) -> TimeoutPhase:
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect"
    if isinstance(exc, httpx.ReadTimeout):
        return "read"
    if isinstance(exc, httpx.WriteTimeout):
        return "write"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool"
    if isinstance(exc, httpx.TimeoutException):
        return "total"
    if isinstance(exc, TimeoutError):
        return "operation"

    exc_type_name = exc.__class__.__name__.casefold()
    if "apitimeouterror" in exc_type_name or "timeout" in exc_type_name:
        return "operation"
    return "unknown"


def build_timeout_failure(
    policy: TicketAgentTimeoutPolicy,
    *,
    phase: TimeoutPhase,
    elapsed_ms: float | None = None,
    idempotency_key_present: bool = False,
) -> TimeoutFailure:
    retryable = is_timeout_retry_allowed(
        policy,
        idempotency_key_present=idempotency_key_present,
    )
    return TimeoutFailure(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        phase=phase,
        error_code=policy.error_code,
        status_code=policy.status_code,
        retryable=retryable,
        fallback_allowed=policy.fallback_allowed,
        recovery_action=policy.recovery_action,
        user_message=policy.user_message,
        elapsed_ms=elapsed_ms,
    )


def is_timeout_retry_allowed(
    policy: TicketAgentTimeoutPolicy,
    *,
    idempotency_key_present: bool = False,
) -> bool:
    if not policy.retryable or policy.max_retries <= 0:
        return False
    if policy.requires_idempotency_key and not idempotency_key_present:
        return False
    return True


def sanitize_timeout_metric_attributes(
    attributes: dict[str, object],
) -> dict[str, str | int | float | bool]:
    safe_attributes: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in TIMEOUT_HIGH_CARDINALITY_ATTRIBUTE_KEYS:
            continue
        normalized_value = _normalize_attribute_value(value)
        if normalized_value is not None:
            safe_attributes[str(key)] = normalized_value
    return safe_attributes


def _validate_positive_timeout(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")


def _round_timeout(value: float) -> float:
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
