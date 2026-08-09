from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ProtectedDependencyKind = Literal[
    "llm",
    "embedding",
    "java_read_tool",
    "java_write_tool",
    "vector_store",
    "milvus",
    "rag_generation",
]
RateLimitScope = Literal[
    "global",
    "per_user",
    "per_tenant",
    "per_dependency",
    "per_operation",
]
RateLimitDecisionReason = Literal[
    "within_limit",
    "near_limit",
    "limit_exceeded",
]
CircuitBreakerState = Literal["closed", "open", "half_open"]
CircuitBreakerDecisionReason = Literal[
    "closed_allows_call",
    "failure_threshold_reached",
    "open_fast_fail",
    "open_allows_half_open_probe",
    "half_open_allows_probe",
    "half_open_probe_limit_reached",
]
CircuitBreakerResultReason = Literal[
    "closed_success_recorded",
    "closed_failure_recorded",
    "opened_after_failure_threshold",
    "half_open_success_recorded",
    "closed_after_half_open_successes",
    "reopened_after_half_open_failure",
]
DegradationTrigger = Literal[
    "none",
    "rate_limited",
    "circuit_open",
    "circuit_half_open_busy",
    "timeout",
    "retry_exhausted",
    "dependency_unavailable",
]
DegradationMode = Literal[
    "none",
    "return_safe_fallback",
    "use_cache_or_return_no_context",
    "return_no_context",
    "retry_later",
    "require_manual_review",
]
ProtectionAction = Literal[
    "allow",
    "allow_probe",
    "throttle",
    "fail_fast",
]
ProtectionDecisionReason = Literal[
    "allowed",
    "allowed_near_rate_limit",
    "rate_limit_exceeded",
    "circuit_open",
    "half_open_probe_limit_reached",
    "half_open_probe_allowed",
]

DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60.0
DEFAULT_RATE_LIMIT_NEAR_LIMIT_RATIO = 0.8
DEFAULT_CIRCUIT_ROLLING_WINDOW_SECONDS = 60.0
DEFAULT_CIRCUIT_OPEN_SECONDS = 30.0
DEFAULT_CIRCUIT_FAILURE_RATE_THRESHOLD = 0.5
DEFAULT_CIRCUIT_FAILURE_COUNT_THRESHOLD = 5
DEFAULT_CIRCUIT_MINIMUM_REQUEST_COUNT = 5
DEFAULT_CIRCUIT_HALF_OPEN_MAX_PROBES = 1
DEFAULT_CIRCUIT_HALF_OPEN_SUCCESS_THRESHOLD = 2

RESILIENCE_HIGH_CARDINALITY_ATTRIBUTE_KEYS = frozenset(
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
class RateLimitPolicy:
    dependency_kind: ProtectedDependencyKind
    operation: str
    max_requests: int
    window_seconds: float = DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    scope: RateLimitScope = "per_dependency"
    near_limit_ratio: float = DEFAULT_RATE_LIMIT_NEAR_LIMIT_RATIO
    burst_size: int | None = None
    status_code: int = 429
    user_message: str = "请求过于频繁，请稍后再试。"

    def __post_init__(self) -> None:
        _validate_non_blank_text(self.operation, "operation")
        _validate_positive_int(self.max_requests, "max_requests")
        _validate_positive_number(self.window_seconds, "window_seconds")
        _validate_ratio(self.near_limit_ratio, "near_limit_ratio")
        _validate_http_error_status(self.status_code, "status_code")
        if self.burst_size is not None:
            _validate_positive_int(self.burst_size, "burst_size")
            if self.burst_size > self.max_requests:
                raise ValueError("burst_size must be less than or equal to max_requests.")

    @property
    def effective_burst_size(self) -> int:
        return self.burst_size or self.max_requests


@dataclass(frozen=True)
class RateLimitUsage:
    requests_in_window: int
    window_seconds_remaining: float

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.requests_in_window, "requests_in_window")
        _validate_non_negative_number(
            self.window_seconds_remaining,
            "window_seconds_remaining",
        )


@dataclass(frozen=True)
class RateLimitDecision:
    dependency_kind: ProtectedDependencyKind
    operation: str
    allowed: bool
    reason: RateLimitDecisionReason
    requests_in_window: int
    max_requests: int
    remaining_requests: int
    retry_after_seconds: float | None
    near_limit: bool
    status_code: int = 429

    def log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "rate_limit_allowed": self.allowed,
            "rate_limit_reason": self.reason,
            "requests_in_window": self.requests_in_window,
            "max_requests": self.max_requests,
            "remaining_requests": self.remaining_requests,
            "near_limit": self.near_limit,
            "status_code": self.status_code,
        }
        if self.retry_after_seconds is not None:
            fields["retry_after_seconds"] = self.retry_after_seconds
        return fields

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        return {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "rate_limit_allowed": self.allowed,
            "rate_limit_reason": self.reason,
            "near_limit": self.near_limit,
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    dependency_kind: ProtectedDependencyKind
    operation: str
    failure_count_threshold: int = DEFAULT_CIRCUIT_FAILURE_COUNT_THRESHOLD
    failure_rate_threshold: float = DEFAULT_CIRCUIT_FAILURE_RATE_THRESHOLD
    minimum_request_count: int = DEFAULT_CIRCUIT_MINIMUM_REQUEST_COUNT
    rolling_window_seconds: float = DEFAULT_CIRCUIT_ROLLING_WINDOW_SECONDS
    open_seconds: float = DEFAULT_CIRCUIT_OPEN_SECONDS
    half_open_max_probes: int = DEFAULT_CIRCUIT_HALF_OPEN_MAX_PROBES
    half_open_success_threshold: int = DEFAULT_CIRCUIT_HALF_OPEN_SUCCESS_THRESHOLD

    def __post_init__(self) -> None:
        _validate_non_blank_text(self.operation, "operation")
        _validate_positive_int(self.failure_count_threshold, "failure_count_threshold")
        _validate_ratio(self.failure_rate_threshold, "failure_rate_threshold")
        _validate_positive_int(self.minimum_request_count, "minimum_request_count")
        _validate_positive_number(
            self.rolling_window_seconds,
            "rolling_window_seconds",
        )
        _validate_positive_number(self.open_seconds, "open_seconds")
        _validate_positive_int(self.half_open_max_probes, "half_open_max_probes")
        _validate_positive_int(
            self.half_open_success_threshold,
            "half_open_success_threshold",
        )


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    state: CircuitBreakerState
    request_count: int = 0
    failure_count: int = 0
    consecutive_success_count: int = 0
    open_elapsed_seconds: float = 0.0
    half_open_in_flight_probes: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.request_count, "request_count")
        _validate_non_negative_int(self.failure_count, "failure_count")
        _validate_non_negative_int(
            self.consecutive_success_count,
            "consecutive_success_count",
        )
        _validate_non_negative_number(self.open_elapsed_seconds, "open_elapsed_seconds")
        _validate_non_negative_int(
            self.half_open_in_flight_probes,
            "half_open_in_flight_probes",
        )
        if self.failure_count > self.request_count and self.state == "closed":
            raise ValueError("failure_count must not exceed request_count in closed state.")

    @property
    def failure_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return _round_ratio(self.failure_count / self.request_count)


@dataclass(frozen=True)
class CircuitBreakerDecision:
    dependency_kind: ProtectedDependencyKind
    operation: str
    state: CircuitBreakerState
    allow_call: bool
    reason: CircuitBreakerDecisionReason
    next_state: CircuitBreakerState
    failure_rate: float
    remaining_open_seconds: float | None = None
    half_open_probe_allowed: bool = False

    def log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "circuit_state": self.state,
            "circuit_allow_call": self.allow_call,
            "circuit_reason": self.reason,
            "next_circuit_state": self.next_state,
            "failure_rate": self.failure_rate,
            "half_open_probe_allowed": self.half_open_probe_allowed,
        }
        if self.remaining_open_seconds is not None:
            fields["remaining_open_seconds"] = self.remaining_open_seconds
        return fields

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        return {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "circuit_state": self.state,
            "circuit_allow_call": self.allow_call,
            "circuit_reason": self.reason,
            "next_circuit_state": self.next_state,
        }


@dataclass(frozen=True)
class CircuitBreakerResult:
    dependency_kind: ProtectedDependencyKind
    operation: str
    call_succeeded: bool
    previous_state: CircuitBreakerState
    next_snapshot: CircuitBreakerSnapshot
    reason: CircuitBreakerResultReason


@dataclass(frozen=True)
class DegradationPlan:
    dependency_kind: ProtectedDependencyKind
    operation: str
    trigger: DegradationTrigger
    mode: DegradationMode
    should_call_dependency: bool
    should_retry: bool
    should_use_cache: bool
    should_call_model: bool
    user_message: str
    reason: str
    status_code: int = 503

    def log_fields(self) -> dict[str, str | int | float | bool]:
        return {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "degradation_trigger": self.trigger,
            "degradation_mode": self.mode,
            "should_call_dependency": self.should_call_dependency,
            "should_retry": self.should_retry,
            "should_use_cache": self.should_use_cache,
            "should_call_model": self.should_call_model,
            "status_code": self.status_code,
        }

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        return {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "degradation_trigger": self.trigger,
            "degradation_mode": self.mode,
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class TicketAgentResiliencePolicy:
    dependency_kind: ProtectedDependencyKind
    operation: str
    rate_limit: RateLimitPolicy
    circuit_breaker: CircuitBreakerPolicy
    degradation_mode: DegradationMode
    fallback_allowed: bool
    cost_sensitive: bool = False
    user_message: str = "外部依赖暂时不可用，系统已启用保护策略。"

    def __post_init__(self) -> None:
        _validate_non_blank_text(self.operation, "operation")
        if self.rate_limit.dependency_kind != self.dependency_kind:
            raise ValueError("rate_limit dependency_kind must match policy.")
        if self.circuit_breaker.dependency_kind != self.dependency_kind:
            raise ValueError("circuit_breaker dependency_kind must match policy.")
        if self.rate_limit.operation != self.operation:
            raise ValueError("rate_limit operation must match policy.")
        if self.circuit_breaker.operation != self.operation:
            raise ValueError("circuit_breaker operation must match policy.")


@dataclass(frozen=True)
class DependencyProtectionDecision:
    dependency_kind: ProtectedDependencyKind
    operation: str
    allowed: bool
    action: ProtectionAction
    reason: ProtectionDecisionReason
    rate_limit_decision: RateLimitDecision
    circuit_breaker_decision: CircuitBreakerDecision | None
    degradation_plan: DegradationPlan
    cost_sensitive: bool = False

    def log_fields(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "protection_allowed": self.allowed,
            "protection_action": self.action,
            "protection_reason": self.reason,
            "cost_sensitive": self.cost_sensitive,
        }
        fields.update(self.rate_limit_decision.log_fields())
        if self.circuit_breaker_decision is not None:
            fields.update(self.circuit_breaker_decision.log_fields())
        fields.update(self.degradation_plan.log_fields())
        return fields

    def metric_attributes(self) -> dict[str, str | int | float | bool]:
        fields: dict[str, str | int | float | bool] = {
            "dependency_kind": self.dependency_kind,
            "operation": self.operation,
            "protection_allowed": self.allowed,
            "protection_action": self.action,
            "protection_reason": self.reason,
            "cost_sensitive": self.cost_sensitive,
        }
        fields.update(self.degradation_plan.metric_attributes())
        return fields


def build_ticket_agent_resilience_policies() -> dict[str, TicketAgentResiliencePolicy]:
    return {
        "llm.intent_classification": _build_policy(
            dependency_kind="llm",
            operation="ticket_intent_classification",
            max_requests=60,
            degradation_mode="return_safe_fallback",
            fallback_allowed=True,
            cost_sensitive=True,
            user_message="模型意图识别暂时受保护限制，系统会使用安全兜底策略。",
        ),
        "llm.field_extraction": _build_policy(
            dependency_kind="llm",
            operation="ticket_field_extraction",
            max_requests=60,
            degradation_mode="return_safe_fallback",
            fallback_allowed=True,
            cost_sensitive=True,
            user_message="模型字段提取暂时受保护限制，系统会使用安全兜底策略。",
        ),
        "embedding.create": _build_policy(
            dependency_kind="embedding",
            operation="create_embeddings",
            max_requests=30,
            degradation_mode="retry_later",
            fallback_allowed=False,
            cost_sensitive=True,
            user_message="向量生成暂时受保护限制，请稍后重试。",
        ),
        "java.query_order": _build_policy(
            dependency_kind="java_read_tool",
            operation="query_order",
            max_requests=120,
            degradation_mode="retry_later",
            fallback_allowed=True,
            user_message="订单查询工具暂时繁忙，请稍后重试。",
        ),
        "java.create_ticket": _build_policy(
            dependency_kind="java_write_tool",
            operation="create_ticket",
            max_requests=30,
            degradation_mode="require_manual_review",
            fallback_allowed=False,
            user_message="创建工单工具暂时受保护限制，请稍后确认是否已创建成功。",
        ),
        "qdrant.vector_search": _build_policy(
            dependency_kind="vector_store",
            operation="vector_search",
            max_requests=120,
            degradation_mode="use_cache_or_return_no_context",
            fallback_allowed=True,
            user_message="Qdrant 检索暂时受保护限制，系统会尝试使用缓存或返回无上下文兜底。",
        ),
        "milvus.vector_search": _build_policy(
            dependency_kind="milvus",
            operation="vector_search",
            max_requests=120,
            degradation_mode="use_cache_or_return_no_context",
            fallback_allowed=True,
            user_message="Milvus 检索暂时受保护限制，系统会尝试使用缓存或返回无上下文兜底。",
        ),
        "rag.generate_answer": _build_policy(
            dependency_kind="rag_generation",
            operation="generate_answer",
            max_requests=60,
            degradation_mode="return_safe_fallback",
            fallback_allowed=True,
            cost_sensitive=True,
            user_message="RAG 回答生成暂时受保护限制，系统会返回安全兜底答案。",
        ),
    }


def decide_rate_limit(
    policy: RateLimitPolicy,
    usage: RateLimitUsage,
) -> RateLimitDecision:
    if usage.requests_in_window >= policy.max_requests:
        return RateLimitDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            allowed=False,
            reason="limit_exceeded",
            requests_in_window=usage.requests_in_window,
            max_requests=policy.max_requests,
            remaining_requests=0,
            retry_after_seconds=_round_seconds(usage.window_seconds_remaining),
            near_limit=True,
            status_code=policy.status_code,
        )

    remaining = policy.max_requests - usage.requests_in_window - 1
    next_count = usage.requests_in_window + 1
    near_limit = next_count >= math.ceil(
        policy.max_requests * policy.near_limit_ratio
    )
    return RateLimitDecision(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        allowed=True,
        reason="near_limit" if near_limit else "within_limit",
        requests_in_window=usage.requests_in_window,
        max_requests=policy.max_requests,
        remaining_requests=max(remaining, 0),
        retry_after_seconds=None,
        near_limit=near_limit,
        status_code=policy.status_code,
    )


def decide_circuit_breaker(
    policy: CircuitBreakerPolicy,
    snapshot: CircuitBreakerSnapshot,
) -> CircuitBreakerDecision:
    if snapshot.state == "closed":
        if _should_open_circuit(policy, snapshot):
            return CircuitBreakerDecision(
                dependency_kind=policy.dependency_kind,
                operation=policy.operation,
                state=snapshot.state,
                allow_call=False,
                reason="failure_threshold_reached",
                next_state="open",
                failure_rate=snapshot.failure_rate,
            )
        return CircuitBreakerDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            state=snapshot.state,
            allow_call=True,
            reason="closed_allows_call",
            next_state="closed",
            failure_rate=snapshot.failure_rate,
        )

    if snapshot.state == "open":
        remaining = policy.open_seconds - snapshot.open_elapsed_seconds
        if remaining > 0:
            return CircuitBreakerDecision(
                dependency_kind=policy.dependency_kind,
                operation=policy.operation,
                state=snapshot.state,
                allow_call=False,
                reason="open_fast_fail",
                next_state="open",
                failure_rate=snapshot.failure_rate,
                remaining_open_seconds=_round_seconds(remaining),
            )
        return CircuitBreakerDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            state=snapshot.state,
            allow_call=True,
            reason="open_allows_half_open_probe",
            next_state="half_open",
            failure_rate=snapshot.failure_rate,
            remaining_open_seconds=0.0,
            half_open_probe_allowed=True,
        )

    if snapshot.half_open_in_flight_probes >= policy.half_open_max_probes:
        return CircuitBreakerDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            state=snapshot.state,
            allow_call=False,
            reason="half_open_probe_limit_reached",
            next_state="half_open",
            failure_rate=snapshot.failure_rate,
        )
    return CircuitBreakerDecision(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        state=snapshot.state,
        allow_call=True,
        reason="half_open_allows_probe",
        next_state="half_open",
        failure_rate=snapshot.failure_rate,
        half_open_probe_allowed=True,
    )


def record_circuit_breaker_result(
    policy: CircuitBreakerPolicy,
    snapshot: CircuitBreakerSnapshot,
    *,
    call_succeeded: bool,
) -> CircuitBreakerResult:
    if snapshot.state == "closed":
        next_request_count = snapshot.request_count + 1
        next_failure_count = snapshot.failure_count + (0 if call_succeeded else 1)
        next_snapshot = CircuitBreakerSnapshot(
            state="closed",
            request_count=next_request_count,
            failure_count=next_failure_count,
            consecutive_success_count=snapshot.consecutive_success_count + 1
            if call_succeeded
            else 0,
        )
        if not call_succeeded and _should_open_circuit(policy, next_snapshot):
            next_snapshot = CircuitBreakerSnapshot(state="open")
            reason: CircuitBreakerResultReason = "opened_after_failure_threshold"
        else:
            reason = (
                "closed_success_recorded"
                if call_succeeded
                else "closed_failure_recorded"
            )
        return CircuitBreakerResult(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            call_succeeded=call_succeeded,
            previous_state=snapshot.state,
            next_snapshot=next_snapshot,
            reason=reason,
        )

    if snapshot.state == "half_open":
        if not call_succeeded:
            return CircuitBreakerResult(
                dependency_kind=policy.dependency_kind,
                operation=policy.operation,
                call_succeeded=False,
                previous_state=snapshot.state,
                next_snapshot=CircuitBreakerSnapshot(state="open"),
                reason="reopened_after_half_open_failure",
            )
        next_success_count = snapshot.consecutive_success_count + 1
        if next_success_count >= policy.half_open_success_threshold:
            next_snapshot = CircuitBreakerSnapshot(state="closed")
            reason = "closed_after_half_open_successes"
        else:
            next_snapshot = CircuitBreakerSnapshot(
                state="half_open",
                consecutive_success_count=next_success_count,
            )
            reason = "half_open_success_recorded"
        return CircuitBreakerResult(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            call_succeeded=True,
            previous_state=snapshot.state,
            next_snapshot=next_snapshot,
            reason=reason,
        )

    if call_succeeded:
        next_snapshot = CircuitBreakerSnapshot(state="half_open", consecutive_success_count=1)
        reason = "half_open_success_recorded"
    else:
        next_snapshot = CircuitBreakerSnapshot(state="open")
        reason = "reopened_after_half_open_failure"
    return CircuitBreakerResult(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        call_succeeded=call_succeeded,
        previous_state=snapshot.state,
        next_snapshot=next_snapshot,
        reason=reason,
    )


def build_degradation_plan(
    policy: TicketAgentResiliencePolicy,
    *,
    trigger: DegradationTrigger,
    has_cached_result: bool = False,
    has_safe_context: bool = False,
) -> DegradationPlan:
    if trigger == "none":
        return DegradationPlan(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            trigger="none",
            mode="none",
            should_call_dependency=True,
            should_retry=False,
            should_use_cache=False,
            should_call_model=policy.dependency_kind != "rag_generation",
            user_message="保护策略允许正常调用。",
            reason="当前未触发限流、熔断或降级。",
            status_code=200,
        )

    if policy.degradation_mode == "use_cache_or_return_no_context":
        if has_cached_result:
            return DegradationPlan(
                dependency_kind=policy.dependency_kind,
                operation=policy.operation,
                trigger=trigger,
                mode="use_cache_or_return_no_context",
                should_call_dependency=False,
                should_retry=False,
                should_use_cache=True,
                should_call_model=True,
                user_message="当前检索依赖受保护限制，系统会使用最近一次可用缓存生成回答。",
                reason="检索依赖不可用，但存在可用缓存。",
                status_code=200,
            )
        return DegradationPlan(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            trigger=trigger,
            mode="return_no_context",
            should_call_dependency=False,
            should_retry=False,
            should_use_cache=False,
            should_call_model=False,
            user_message="当前知识库检索服务暂时不可用，无法根据知识库上下文回答。",
            reason="检索依赖不可用，且没有可用缓存。",
            status_code=503,
        )

    if policy.degradation_mode == "return_safe_fallback":
        return DegradationPlan(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            trigger=trigger,
            mode="return_safe_fallback",
            should_call_dependency=False,
            should_retry=False,
            should_use_cache=False,
            should_call_model=False,
            user_message=policy.user_message,
            reason="依赖受保护限制，返回安全兜底结果。",
            status_code=200 if has_safe_context else 503,
        )

    if policy.degradation_mode == "require_manual_review":
        return DegradationPlan(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            trigger=trigger,
            mode="require_manual_review",
            should_call_dependency=False,
            should_retry=False,
            should_use_cache=False,
            should_call_model=False,
            user_message=policy.user_message,
            reason="写操作受保护限制，不能自动降级为重复写入。",
            status_code=503,
        )

    return DegradationPlan(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        trigger=trigger,
        mode="retry_later",
        should_call_dependency=False,
        should_retry=False,
        should_use_cache=False,
        should_call_model=False,
        user_message=policy.user_message,
        reason="依赖受保护限制，请稍后重试。",
        status_code=503,
    )


def evaluate_dependency_protection(
    policy: TicketAgentResiliencePolicy,
    *,
    rate_limit_usage: RateLimitUsage,
    circuit_breaker_snapshot: CircuitBreakerSnapshot,
    has_cached_result: bool = False,
    has_safe_context: bool = False,
) -> DependencyProtectionDecision:
    rate_limit_decision = decide_rate_limit(policy.rate_limit, rate_limit_usage)
    if not rate_limit_decision.allowed:
        degradation_plan = build_degradation_plan(
            policy,
            trigger="rate_limited",
            has_cached_result=has_cached_result,
            has_safe_context=has_safe_context,
        )
        return DependencyProtectionDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            allowed=False,
            action="throttle",
            reason="rate_limit_exceeded",
            rate_limit_decision=rate_limit_decision,
            circuit_breaker_decision=None,
            degradation_plan=degradation_plan,
            cost_sensitive=policy.cost_sensitive,
        )

    circuit_breaker_decision = decide_circuit_breaker(
        policy.circuit_breaker,
        circuit_breaker_snapshot,
    )
    if not circuit_breaker_decision.allow_call:
        trigger: DegradationTrigger = (
            "circuit_half_open_busy"
            if circuit_breaker_decision.reason == "half_open_probe_limit_reached"
            else "circuit_open"
        )
        degradation_plan = build_degradation_plan(
            policy,
            trigger=trigger,
            has_cached_result=has_cached_result,
            has_safe_context=has_safe_context,
        )
        return DependencyProtectionDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            allowed=False,
            action="fail_fast",
            reason="half_open_probe_limit_reached"
            if trigger == "circuit_half_open_busy"
            else "circuit_open",
            rate_limit_decision=rate_limit_decision,
            circuit_breaker_decision=circuit_breaker_decision,
            degradation_plan=degradation_plan,
            cost_sensitive=policy.cost_sensitive,
        )

    degradation_plan = build_degradation_plan(policy, trigger="none")
    if circuit_breaker_decision.half_open_probe_allowed:
        return DependencyProtectionDecision(
            dependency_kind=policy.dependency_kind,
            operation=policy.operation,
            allowed=True,
            action="allow_probe",
            reason="half_open_probe_allowed",
            rate_limit_decision=rate_limit_decision,
            circuit_breaker_decision=circuit_breaker_decision,
            degradation_plan=degradation_plan,
            cost_sensitive=policy.cost_sensitive,
        )

    return DependencyProtectionDecision(
        dependency_kind=policy.dependency_kind,
        operation=policy.operation,
        allowed=True,
        action="allow",
        reason="allowed_near_rate_limit"
        if rate_limit_decision.near_limit
        else "allowed",
        rate_limit_decision=rate_limit_decision,
        circuit_breaker_decision=circuit_breaker_decision,
        degradation_plan=degradation_plan,
        cost_sensitive=policy.cost_sensitive,
    )


def sanitize_resilience_metric_attributes(
    attributes: dict[str, object],
) -> dict[str, str | int | float | bool]:
    safe_attributes: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in RESILIENCE_HIGH_CARDINALITY_ATTRIBUTE_KEYS:
            continue
        normalized_value = _normalize_attribute_value(value)
        if normalized_value is not None:
            safe_attributes[str(key)] = normalized_value
    return safe_attributes


def _build_policy(
    *,
    dependency_kind: ProtectedDependencyKind,
    operation: str,
    max_requests: int,
    degradation_mode: DegradationMode,
    fallback_allowed: bool,
    user_message: str,
    cost_sensitive: bool = False,
) -> TicketAgentResiliencePolicy:
    rate_limit = RateLimitPolicy(
        dependency_kind=dependency_kind,
        operation=operation,
        max_requests=max_requests,
        scope="per_dependency",
        user_message=user_message,
    )
    circuit_breaker = CircuitBreakerPolicy(
        dependency_kind=dependency_kind,
        operation=operation,
    )
    return TicketAgentResiliencePolicy(
        dependency_kind=dependency_kind,
        operation=operation,
        rate_limit=rate_limit,
        circuit_breaker=circuit_breaker,
        degradation_mode=degradation_mode,
        fallback_allowed=fallback_allowed,
        cost_sensitive=cost_sensitive,
        user_message=user_message,
    )


def _should_open_circuit(
    policy: CircuitBreakerPolicy,
    snapshot: CircuitBreakerSnapshot,
) -> bool:
    if snapshot.request_count < policy.minimum_request_count:
        return False
    return (
        snapshot.failure_count >= policy.failure_count_threshold
        and snapshot.failure_rate >= policy.failure_rate_threshold
    )


def _validate_non_blank_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank.")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")


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


def _validate_ratio(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1.")


def _validate_http_error_status(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 400 or value > 599:
        raise ValueError(f"{field_name} must be an HTTP error status.")


def _round_seconds(value: float) -> float:
    return max(0.0, round(value, 3))


def _round_ratio(value: float) -> float:
    return round(value, 6)


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
