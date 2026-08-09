import pytest

from app.agents.resilience_strategy import (
    CircuitBreakerPolicy,
    CircuitBreakerSnapshot,
    RateLimitPolicy,
    RateLimitUsage,
    TicketAgentResiliencePolicy,
    build_degradation_plan,
    build_ticket_agent_resilience_policies,
    decide_circuit_breaker,
    decide_rate_limit,
    evaluate_dependency_protection,
    record_circuit_breaker_result,
    sanitize_resilience_metric_attributes,
)


def test_rate_limit_allows_request_when_window_has_capacity() -> None:
    policy = RateLimitPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        max_requests=10,
    )
    usage = RateLimitUsage(requests_in_window=3, window_seconds_remaining=20)

    decision = decide_rate_limit(policy, usage)

    assert decision.allowed is True
    assert decision.reason == "within_limit"
    assert decision.remaining_requests == 6
    assert decision.retry_after_seconds is None
    assert decision.metric_attributes() == {
        "dependency_kind": "llm",
        "operation": "ticket_intent_classification",
        "rate_limit_allowed": True,
        "rate_limit_reason": "within_limit",
        "near_limit": False,
        "status_code": 429,
    }


def test_rate_limit_marks_near_limit_before_capacity_is_exhausted() -> None:
    policy = RateLimitPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        max_requests=10,
        near_limit_ratio=0.8,
    )
    usage = RateLimitUsage(requests_in_window=7, window_seconds_remaining=15)

    decision = decide_rate_limit(policy, usage)

    assert decision.allowed is True
    assert decision.reason == "near_limit"
    assert decision.remaining_requests == 2
    assert decision.near_limit is True


def test_rate_limit_blocks_request_after_capacity_is_exhausted() -> None:
    policy = RateLimitPolicy(
        dependency_kind="embedding",
        operation="create_embeddings",
        max_requests=5,
    )
    usage = RateLimitUsage(requests_in_window=5, window_seconds_remaining=12.3456)

    decision = decide_rate_limit(policy, usage)

    assert decision.allowed is False
    assert decision.reason == "limit_exceeded"
    assert decision.remaining_requests == 0
    assert decision.retry_after_seconds == 12.346
    assert decision.status_code == 429


def test_rate_limit_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_requests"):
        RateLimitPolicy(
            dependency_kind="llm",
            operation="chat",
            max_requests=0,
        )

    with pytest.raises(ValueError, match="near_limit_ratio"):
        RateLimitPolicy(
            dependency_kind="llm",
            operation="chat",
            max_requests=10,
            near_limit_ratio=0,
        )

    with pytest.raises(ValueError, match="burst_size"):
        RateLimitPolicy(
            dependency_kind="llm",
            operation="chat",
            max_requests=10,
            burst_size=11,
        )


def test_circuit_breaker_closed_state_allows_call_before_threshold() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="vector_store",
        operation="vector_search",
        failure_count_threshold=5,
        failure_rate_threshold=0.5,
        minimum_request_count=5,
    )
    snapshot = CircuitBreakerSnapshot(
        state="closed",
        request_count=4,
        failure_count=4,
    )

    decision = decide_circuit_breaker(policy, snapshot)

    assert decision.allow_call is True
    assert decision.reason == "closed_allows_call"
    assert decision.next_state == "closed"


def test_circuit_breaker_closed_state_opens_after_failure_threshold() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="vector_store",
        operation="vector_search",
        failure_count_threshold=5,
        failure_rate_threshold=0.5,
        minimum_request_count=5,
    )
    snapshot = CircuitBreakerSnapshot(
        state="closed",
        request_count=10,
        failure_count=5,
    )

    decision = decide_circuit_breaker(policy, snapshot)

    assert decision.allow_call is False
    assert decision.reason == "failure_threshold_reached"
    assert decision.next_state == "open"
    assert decision.failure_rate == 0.5


def test_circuit_breaker_open_state_fails_fast_before_cooldown_expires() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        open_seconds=30,
    )
    snapshot = CircuitBreakerSnapshot(
        state="open",
        open_elapsed_seconds=12,
    )

    decision = decide_circuit_breaker(policy, snapshot)

    assert decision.allow_call is False
    assert decision.reason == "open_fast_fail"
    assert decision.remaining_open_seconds == 18
    assert decision.next_state == "open"


def test_circuit_breaker_open_state_allows_half_open_probe_after_cooldown() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        open_seconds=30,
    )
    snapshot = CircuitBreakerSnapshot(
        state="open",
        open_elapsed_seconds=31,
    )

    decision = decide_circuit_breaker(policy, snapshot)

    assert decision.allow_call is True
    assert decision.reason == "open_allows_half_open_probe"
    assert decision.next_state == "half_open"
    assert decision.half_open_probe_allowed is True


def test_circuit_breaker_half_open_limits_concurrent_probe_count() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        half_open_max_probes=1,
    )
    snapshot = CircuitBreakerSnapshot(
        state="half_open",
        half_open_in_flight_probes=1,
    )

    decision = decide_circuit_breaker(policy, snapshot)

    assert decision.allow_call is False
    assert decision.reason == "half_open_probe_limit_reached"
    assert decision.next_state == "half_open"


def test_record_circuit_breaker_result_opens_after_closed_failures() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="vector_store",
        operation="vector_search",
        failure_count_threshold=2,
        failure_rate_threshold=0.5,
        minimum_request_count=2,
    )
    snapshot = CircuitBreakerSnapshot(
        state="closed",
        request_count=1,
        failure_count=1,
    )

    result = record_circuit_breaker_result(
        policy,
        snapshot,
        call_succeeded=False,
    )

    assert result.reason == "opened_after_failure_threshold"
    assert result.next_snapshot.state == "open"


def test_record_circuit_breaker_result_closes_after_half_open_successes() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
        half_open_success_threshold=2,
    )
    snapshot = CircuitBreakerSnapshot(
        state="half_open",
        consecutive_success_count=1,
    )

    result = record_circuit_breaker_result(
        policy,
        snapshot,
        call_succeeded=True,
    )

    assert result.reason == "closed_after_half_open_successes"
    assert result.next_snapshot.state == "closed"


def test_record_circuit_breaker_result_reopens_after_half_open_failure() -> None:
    policy = CircuitBreakerPolicy(
        dependency_kind="llm",
        operation="ticket_intent_classification",
    )
    snapshot = CircuitBreakerSnapshot(state="half_open")

    result = record_circuit_breaker_result(
        policy,
        snapshot,
        call_succeeded=False,
    )

    assert result.reason == "reopened_after_half_open_failure"
    assert result.next_snapshot.state == "open"


def test_build_ticket_agent_resilience_policies_cover_current_dependencies() -> None:
    policies = build_ticket_agent_resilience_policies()

    assert policies["llm.intent_classification"].cost_sensitive is True
    assert policies["embedding.create"].degradation_mode == "retry_later"
    assert policies["java.query_order"].fallback_allowed is True
    assert policies["java.create_ticket"].degradation_mode == "require_manual_review"
    assert policies["qdrant.vector_search"].degradation_mode == (
        "use_cache_or_return_no_context"
    )
    assert policies["milvus.vector_search"].dependency_kind == "milvus"
    assert policies["rag.generate_answer"].cost_sensitive is True


def test_degradation_plan_uses_cache_for_vector_store_when_available() -> None:
    policy = build_ticket_agent_resilience_policies()["qdrant.vector_search"]

    plan = build_degradation_plan(
        policy,
        trigger="circuit_open",
        has_cached_result=True,
    )

    assert plan.mode == "use_cache_or_return_no_context"
    assert plan.should_call_dependency is False
    assert plan.should_use_cache is True
    assert plan.should_call_model is True
    assert plan.status_code == 200


def test_degradation_plan_returns_no_context_without_vector_cache() -> None:
    policy = build_ticket_agent_resilience_policies()["milvus.vector_search"]

    plan = build_degradation_plan(
        policy,
        trigger="rate_limited",
        has_cached_result=False,
    )

    assert plan.mode == "return_no_context"
    assert plan.should_call_dependency is False
    assert plan.should_call_model is False
    assert plan.status_code == 503


def test_degradation_plan_requires_manual_review_for_write_tool() -> None:
    policy = build_ticket_agent_resilience_policies()["java.create_ticket"]

    plan = build_degradation_plan(policy, trigger="circuit_open")

    assert plan.mode == "require_manual_review"
    assert plan.should_call_dependency is False
    assert plan.should_retry is False
    assert plan.should_call_model is False


def test_evaluate_dependency_protection_blocks_by_rate_limit_before_circuit() -> None:
    policy = build_ticket_agent_resilience_policies()["llm.intent_classification"]

    decision = evaluate_dependency_protection(
        policy,
        rate_limit_usage=RateLimitUsage(
            requests_in_window=60,
            window_seconds_remaining=9,
        ),
        circuit_breaker_snapshot=CircuitBreakerSnapshot(
            state="closed",
            request_count=0,
            failure_count=0,
        ),
        has_safe_context=True,
    )

    assert decision.allowed is False
    assert decision.action == "throttle"
    assert decision.reason == "rate_limit_exceeded"
    assert decision.circuit_breaker_decision is None
    assert decision.degradation_plan.mode == "return_safe_fallback"


def test_evaluate_dependency_protection_fails_fast_when_circuit_is_open() -> None:
    policy = build_ticket_agent_resilience_policies()["qdrant.vector_search"]

    decision = evaluate_dependency_protection(
        policy,
        rate_limit_usage=RateLimitUsage(
            requests_in_window=10,
            window_seconds_remaining=20,
        ),
        circuit_breaker_snapshot=CircuitBreakerSnapshot(
            state="open",
            open_elapsed_seconds=1,
        ),
        has_cached_result=True,
    )

    assert decision.allowed is False
    assert decision.action == "fail_fast"
    assert decision.reason == "circuit_open"
    assert decision.circuit_breaker_decision is not None
    assert decision.degradation_plan.should_use_cache is True


def test_evaluate_dependency_protection_allows_half_open_probe() -> None:
    policy = build_ticket_agent_resilience_policies()["llm.intent_classification"]

    decision = evaluate_dependency_protection(
        policy,
        rate_limit_usage=RateLimitUsage(
            requests_in_window=10,
            window_seconds_remaining=20,
        ),
        circuit_breaker_snapshot=CircuitBreakerSnapshot(
            state="open",
            open_elapsed_seconds=31,
        ),
    )

    assert decision.allowed is True
    assert decision.action == "allow_probe"
    assert decision.reason == "half_open_probe_allowed"
    assert decision.degradation_plan.mode == "none"


def test_evaluate_dependency_protection_allows_normal_call_and_marks_near_limit() -> None:
    policy = build_ticket_agent_resilience_policies()["llm.intent_classification"]

    decision = evaluate_dependency_protection(
        policy,
        rate_limit_usage=RateLimitUsage(
            requests_in_window=47,
            window_seconds_remaining=20,
        ),
        circuit_breaker_snapshot=CircuitBreakerSnapshot(state="closed"),
    )

    assert decision.allowed is True
    assert decision.action == "allow"
    assert decision.reason == "allowed_near_rate_limit"
    assert decision.rate_limit_decision.near_limit is True
    assert decision.metric_attributes()["degradation_mode"] == "none"


def test_ticket_agent_resilience_policy_rejects_mismatched_sub_policies() -> None:
    rate_limit = RateLimitPolicy(
        dependency_kind="llm",
        operation="chat",
        max_requests=10,
    )
    circuit_breaker = CircuitBreakerPolicy(
        dependency_kind="embedding",
        operation="chat",
    )

    with pytest.raises(ValueError, match="circuit_breaker dependency_kind"):
        TicketAgentResiliencePolicy(
            dependency_kind="llm",
            operation="chat",
            rate_limit=rate_limit,
            circuit_breaker=circuit_breaker,
            degradation_mode="return_safe_fallback",
            fallback_allowed=True,
        )


def test_sanitize_resilience_metric_attributes_filters_sensitive_high_cardinality() -> None:
    attributes = sanitize_resilience_metric_attributes(
        {
            "dependency_kind": "llm",
            "operation": "chat",
            "trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "span_id": "5fb397be34d26b51",
            "request_id": "req_001",
            "idempotency_key": "ticket-create-001",
            "user_message": "包含手机号的用户原文",
            "prompt": "完整提示词",
            "tool_args": {"order_id": "ORD-001"},
            "protection_action": "fail_fast",
            "failure_rate": 0.6255555,
        }
    )

    assert attributes == {
        "dependency_kind": "llm",
        "operation": "chat",
        "protection_action": "fail_fast",
        "failure_rate": 0.625556,
    }
