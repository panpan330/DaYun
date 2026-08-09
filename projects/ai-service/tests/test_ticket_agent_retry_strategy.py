import httpx
import pytest

from app.agents.retry_strategy import (
    RetryBackoff,
    TicketAgentRetryPolicy,
    build_default_retry_backoff,
    build_ticket_agent_retry_policies,
    classify_error_code_for_retry,
    classify_exception_for_retry,
    classify_http_status_for_retry,
    classify_retry_failure,
    decide_retry,
    sanitize_retry_metric_attributes,
)
from app.core.config import Settings


class FakeRateLimitError(Exception):
    pass


class FakeAPITimeoutError(Exception):
    pass


class FakeAuthenticationError(Exception):
    pass


def make_settings() -> Settings:
    return Settings(
        request_timeout_seconds=30.0,
        llm_max_retries=2,
        java_mock_service_timeout_seconds=5.0,
        qdrant_timeout_seconds=4.0,
        milvus_timeout_seconds=6.0,
        _env_file=None,
    )


def test_retry_backoff_builds_exponential_schedule_with_optional_jitter() -> None:
    backoff = RetryBackoff(
        initial_delay_seconds=0.5,
        multiplier=2,
        max_delay_seconds=2,
        jitter_seconds=0.2,
    )

    assert backoff.build_delay_schedule(4) == [0.5, 1.0, 2.0, 2.0]
    assert backoff.delay_for_retry(1, jitter_ratio=0.5) == 0.6
    assert backoff.delay_for_retry(3, jitter_ratio=0.5) == 2.0


def test_retry_backoff_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="initial_delay_seconds"):
        RetryBackoff(initial_delay_seconds=0)

    with pytest.raises(ValueError, match="multiplier"):
        RetryBackoff(multiplier=0.5)

    with pytest.raises(ValueError, match="max_delay_seconds"):
        RetryBackoff(initial_delay_seconds=2, max_delay_seconds=1)

    with pytest.raises(ValueError, match="jitter_ratio"):
        build_default_retry_backoff().delay_for_retry(1, jitter_ratio=2)


def test_build_ticket_agent_retry_policies_use_existing_settings() -> None:
    policies = build_ticket_agent_retry_policies(make_settings())

    assert policies["llm.intent_classification"].max_retries == 2
    assert policies["llm.intent_classification"].max_attempts == 3
    assert policies["llm.intent_classification"].cost_sensitive is True
    assert policies["embedding.create"].max_retries == 1
    assert policies["java.query_order"].fallback_allowed is True
    assert policies["java.create_ticket"].requires_idempotency_key is True
    assert policies["qdrant.vector_search"].dependency_kind == "vector_store"
    assert policies["milvus.vector_search"].dependency_kind == "milvus"


def test_classify_http_status_for_retry_separates_retryable_and_non_retryable() -> None:
    assert classify_http_status_for_retry(408) == "timeout"
    assert classify_http_status_for_retry(409) == "conflict"
    assert classify_http_status_for_retry(429) == "rate_limited"
    assert classify_http_status_for_retry(500) == "server_error"
    assert classify_http_status_for_retry(503) == "server_error"
    assert classify_http_status_for_retry(400) == "validation_error"
    assert classify_http_status_for_retry(401) == "auth_error"
    assert classify_http_status_for_retry(403) == "permission_error"
    assert classify_http_status_for_retry(404) == "not_found"


def test_classify_error_code_for_retry_maps_local_error_codes() -> None:
    assert classify_error_code_for_retry("LLM_TIMEOUT") == "timeout"
    assert classify_error_code_for_retry("LLM_RATE_LIMIT") == "rate_limited"
    assert classify_error_code_for_retry("NETWORK_ERROR") == "connection_error"
    assert classify_error_code_for_retry("LLM_API_KEY_MISSING") == "auth_error"
    assert classify_error_code_for_retry("TOOL_VALIDATION_ERROR") == (
        "validation_error"
    )
    assert classify_error_code_for_retry("BUSINESS_RULE_REJECTED") == "business_rule"


def test_classify_exception_for_retry_handles_httpx_and_openai_like_names() -> None:
    request = httpx.Request("GET", "http://example.test")

    assert classify_exception_for_retry(
        httpx.ConnectTimeout("connect timeout", request=request)
    ) == "timeout"
    assert classify_exception_for_retry(
        httpx.ConnectError("connection error", request=request)
    ) == "connection_error"
    assert classify_exception_for_retry(FakeRateLimitError("rate limited")) == (
        "rate_limited"
    )
    assert classify_exception_for_retry(FakeAPITimeoutError("timeout")) == "timeout"
    assert classify_exception_for_retry(FakeAuthenticationError("auth")) == (
        "auth_error"
    )
    assert classify_exception_for_retry(RuntimeError("boom")) == "unknown"


def test_classify_retry_failure_prefers_status_then_error_code_then_exception() -> None:
    request = httpx.Request("GET", "http://example.test")

    assert classify_retry_failure(
        status_code=400,
        error_code="LLM_TIMEOUT",
        exc=httpx.ConnectError("connection error", request=request),
    ) == "validation_error"
    assert classify_retry_failure(error_code="LLM_TIMEOUT") == "timeout"
    assert classify_retry_failure(
        exc=httpx.ConnectError("connection error", request=request)
    ) == "connection_error"
    assert classify_retry_failure() == "unknown"


def test_decide_retry_allows_retryable_llm_failure_before_attempts_exhausted() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())[
        "llm.intent_classification"
    ]

    decision = decide_retry(
        policy,
        attempt_number=1,
        failure_category="timeout",
        status_code=504,
        jitter_ratio=0.5,
    )

    assert decision.should_retry is True
    assert decision.reason == "retry_allowed"
    assert decision.next_attempt_number == 2
    assert decision.next_delay_seconds == 0.3
    assert decision.max_attempts == 3
    assert decision.cost_sensitive is True
    assert decision.log_fields()["failure_category"] == "timeout"


def test_decide_retry_uses_retry_after_for_rate_limit_with_cap() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())[
        "llm.intent_classification"
    ]

    decision = decide_retry(
        policy,
        attempt_number=1,
        failure_category="rate_limited",
        status_code=429,
        retry_after_seconds=30,
    )

    assert decision.should_retry is True
    assert decision.reason == "retry_after_allowed"
    assert decision.next_delay_seconds == policy.backoff.max_delay_seconds


def test_decide_retry_blocks_when_max_attempts_are_exhausted() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())[
        "llm.intent_classification"
    ]

    decision = decide_retry(
        policy,
        attempt_number=3,
        failure_category="server_error",
        status_code=500,
    )

    assert decision.should_retry is False
    assert decision.reason == "max_retries_exhausted"
    assert decision.next_attempt_number is None


def test_decide_retry_blocks_non_retryable_failure_category() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())["java.query_order"]

    decision = decide_retry(
        policy,
        attempt_number=1,
        failure_category="validation_error",
        status_code=400,
    )

    assert decision.should_retry is False
    assert decision.reason == "failure_not_retryable"
    assert decision.fallback_allowed is True


def test_decide_retry_blocks_retryable_category_with_non_retryable_status() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())["java.query_order"]

    decision = decide_retry(
        policy,
        attempt_number=1,
        failure_category="server_error",
        status_code=418,
    )

    assert decision.should_retry is False
    assert decision.reason == "status_not_retryable"


def test_write_tool_retry_requires_idempotency_key() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())["java.create_ticket"]

    without_key = decide_retry(
        policy,
        attempt_number=1,
        failure_category="timeout",
        status_code=504,
        idempotency_key_present=False,
    )
    with_key = decide_retry(
        policy,
        attempt_number=1,
        failure_category="timeout",
        status_code=504,
        idempotency_key_present=True,
    )

    assert without_key.should_retry is False
    assert without_key.reason == "idempotency_key_required"
    assert without_key.blocked_by_idempotency is True
    assert with_key.should_retry is True
    assert with_key.next_attempt_number == 2


def test_retry_decision_metric_attributes_are_low_cardinality() -> None:
    policy = build_ticket_agent_retry_policies(make_settings())["java.query_order"]

    decision = decide_retry(
        policy,
        attempt_number=1,
        failure_category="connection_error",
    )

    assert decision.metric_attributes() == {
        "dependency_kind": "java_read_tool",
        "operation": "query_order",
        "failure_category": "connection_error",
        "should_retry": True,
        "retry_decision_reason": "retry_allowed",
        "fallback_allowed": True,
        "blocked_by_idempotency": False,
        "cost_sensitive": False,
    }


def test_sanitize_retry_metric_attributes_excludes_payload_and_request_ids() -> None:
    attributes = sanitize_retry_metric_attributes(
        {
            "dependency_kind": "llm",
            "operation": "chat",
            "trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "request_id": "req_001",
            "idempotency_key": "ticket-create-001",
            "tool_args": {"order_id": "ORD-001"},
            "user_message": "包含手机号的用户原文",
            "should_retry": True,
            "next_delay_seconds": 0.25,
        }
    )

    assert attributes == {
        "dependency_kind": "llm",
        "operation": "chat",
        "should_retry": True,
        "next_delay_seconds": 0.25,
    }


def test_retry_policy_rejects_invalid_identity_status_and_empty_categories() -> None:
    with pytest.raises(ValueError, match="operation"):
        TicketAgentRetryPolicy(
            dependency_kind="llm",
            operation=" ",
            max_retries=1,
            backoff=build_default_retry_backoff(),
            retryable_categories=frozenset({"timeout"}),
            retryable_status_codes=frozenset({504}),
            fallback_allowed=True,
            user_message="retry",
        )

    with pytest.raises(ValueError, match="retryable_categories"):
        TicketAgentRetryPolicy(
            dependency_kind="llm",
            operation="chat",
            max_retries=1,
            backoff=build_default_retry_backoff(),
            retryable_categories=frozenset(),
            retryable_status_codes=frozenset({504}),
            fallback_allowed=True,
            user_message="retry",
        )

    with pytest.raises(ValueError, match="status_code"):
        TicketAgentRetryPolicy(
            dependency_kind="llm",
            operation="chat",
            max_retries=1,
            backoff=build_default_retry_backoff(),
            retryable_categories=frozenset({"timeout"}),
            retryable_status_codes=frozenset({999}),
            fallback_allowed=True,
            user_message="retry",
        )
