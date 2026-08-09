import httpx
import pytest

from app.agents.timeout_strategy import (
    TimeoutBudget,
    TicketAgentTimeoutPolicy,
    build_ticket_agent_timeout_policies,
    build_timeout_budget,
    build_timeout_failure,
    classify_timeout_phase,
    is_timeout_retry_allowed,
    sanitize_timeout_metric_attributes,
)
from app.core.config import Settings


class FakeAPITimeoutError(Exception):
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


def test_timeout_budget_derives_connect_write_pool_and_httpx_kwargs() -> None:
    budget = build_timeout_budget(10.0)

    assert budget.total_seconds == 10.0
    assert budget.resolved_connect_seconds() == 2.0
    assert budget.resolved_read_seconds() == 10.0
    assert budget.resolved_write_seconds() == 4.0
    assert budget.resolved_pool_seconds() == 1.0
    assert budget.as_httpx_timeout_kwargs() == {
        "timeout": 10.0,
        "connect": 2.0,
        "read": 10.0,
        "write": 4.0,
        "pool": 1.0,
    }


def test_timeout_budget_accepts_explicit_phase_budgets() -> None:
    budget = TimeoutBudget(
        total_seconds=20,
        connect_seconds=3,
        read_seconds=15,
        write_seconds=5,
        pool_seconds=2,
    )

    assert budget.as_httpx_timeout_kwargs() == {
        "timeout": 20,
        "connect": 3,
        "read": 15,
        "write": 5,
        "pool": 2,
    }


def test_timeout_budget_rejects_zero_negative_or_non_finite_values() -> None:
    for value in (0, -1, float("inf")):
        with pytest.raises(ValueError, match="greater than 0"):
            TimeoutBudget(total_seconds=value)

    with pytest.raises(ValueError, match="connect_seconds"):
        TimeoutBudget(total_seconds=5, connect_seconds=0)


def test_build_ticket_agent_timeout_policies_use_existing_settings() -> None:
    policies = build_ticket_agent_timeout_policies(make_settings())

    assert policies["llm.intent_classification"].budget.total_seconds == 30.0
    assert policies["llm.intent_classification"].error_code == "LLM_TIMEOUT"
    assert policies["llm.intent_classification"].max_retries == 2
    assert policies["java.query_order"].budget.total_seconds == 5.0
    assert policies["java.query_order"].error_code == "TOOL_TIMEOUT"
    assert policies["qdrant.vector_search"].budget.total_seconds == 4.0
    assert policies["milvus.vector_search"].budget.total_seconds == 6.0


def test_write_tool_timeout_requires_idempotency_before_retry() -> None:
    policy = build_ticket_agent_timeout_policies(make_settings())["java.create_ticket"]

    assert policy.dependency_kind == "java_write_tool"
    assert policy.requires_idempotency_key is True
    assert policy.recovery_action == "require_idempotency_before_retry"
    assert is_timeout_retry_allowed(policy, idempotency_key_present=False) is False
    assert is_timeout_retry_allowed(policy, idempotency_key_present=True) is True


def test_read_tool_timeout_can_retry_without_idempotency_key() -> None:
    policy = build_ticket_agent_timeout_policies(make_settings())["java.query_order"]

    assert policy.dependency_kind == "java_read_tool"
    assert policy.requires_idempotency_key is False
    assert is_timeout_retry_allowed(policy, idempotency_key_present=False) is True


def test_classify_timeout_phase_for_httpx_timeout_subclasses() -> None:
    request = httpx.Request("GET", "http://example.test")

    assert classify_timeout_phase(httpx.ConnectTimeout("connect", request=request)) == (
        "connect"
    )
    assert classify_timeout_phase(httpx.ReadTimeout("read", request=request)) == "read"
    assert classify_timeout_phase(httpx.WriteTimeout("write", request=request)) == (
        "write"
    )
    assert classify_timeout_phase(httpx.PoolTimeout("pool", request=request)) == "pool"
    assert classify_timeout_phase(httpx.TimeoutException("timeout", request=request)) == (
        "total"
    )


def test_classify_timeout_phase_for_operation_level_timeouts() -> None:
    assert classify_timeout_phase(TimeoutError("operation timeout")) == "operation"
    assert classify_timeout_phase(FakeAPITimeoutError("api timeout")) == "operation"
    assert classify_timeout_phase(RuntimeError("boom")) == "unknown"


def test_build_timeout_failure_uses_policy_and_elapsed_ms() -> None:
    policy = build_ticket_agent_timeout_policies(make_settings())[
        "llm.intent_classification"
    ]

    failure = build_timeout_failure(
        policy,
        phase="operation",
        elapsed_ms=30001.129,
    )

    assert failure.dependency_kind == "llm"
    assert failure.operation == "ticket_intent_classification"
    assert failure.phase == "operation"
    assert failure.error_code == "LLM_TIMEOUT"
    assert failure.status_code == 504
    assert failure.retryable is True
    assert failure.fallback_allowed is True
    assert failure.log_fields()["elapsed_ms"] == 30001.13


def test_build_timeout_failure_blocks_write_retry_without_idempotency_key() -> None:
    policy = build_ticket_agent_timeout_policies(make_settings())["java.create_ticket"]

    without_key = build_timeout_failure(
        policy,
        phase="read",
        elapsed_ms=5000,
        idempotency_key_present=False,
    )
    with_key = build_timeout_failure(
        policy,
        phase="read",
        elapsed_ms=5000,
        idempotency_key_present=True,
    )

    assert without_key.retryable is False
    assert with_key.retryable is True
    assert with_key.recovery_action == "require_idempotency_before_retry"


def test_vector_store_timeout_failure_allows_cache_or_no_context_fallback() -> None:
    policy = build_ticket_agent_timeout_policies(make_settings())[
        "qdrant.vector_search"
    ]

    failure = build_timeout_failure(policy, phase="read")

    assert failure.error_code == "RAG_VECTOR_STORE_TIMEOUT"
    assert failure.fallback_allowed is True
    assert failure.recovery_action == "use_cache_or_return_no_context"
    assert failure.user_message == (
        "知识库检索响应超时，系统会尝试使用缓存或返回安全兜底答案。"
    )


def test_timeout_metric_attributes_exclude_request_level_and_payload_fields() -> None:
    attributes = sanitize_timeout_metric_attributes(
        {
            "dependency_kind": "llm",
            "operation": "chat",
            "trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "span_id": "5fb397be34d26b51",
            "thread_id": "ticket-thread-001",
            "user_message": "包含手机号的用户原文",
            "prompt": "完整提示词",
            "status": "timeout",
        }
    )

    assert attributes == {
        "dependency_kind": "llm",
        "operation": "chat",
        "status": "timeout",
    }


def test_timeout_policy_rejects_invalid_identity_and_status() -> None:
    with pytest.raises(ValueError, match="operation"):
        TicketAgentTimeoutPolicy(
            dependency_kind="llm",
            operation=" ",
            budget=build_timeout_budget(5),
            error_code="LLM_TIMEOUT",
            status_code=504,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="return_safe_fallback",
        )

    with pytest.raises(ValueError, match="status_code"):
        TicketAgentTimeoutPolicy(
            dependency_kind="llm",
            operation="chat",
            budget=build_timeout_budget(5),
            error_code="LLM_TIMEOUT",
            status_code=200,
            retryable=True,
            max_retries=1,
            fallback_allowed=True,
            recovery_action="return_safe_fallback",
        )
