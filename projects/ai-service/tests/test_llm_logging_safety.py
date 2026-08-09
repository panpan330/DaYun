from app.core.llm_logging_safety import (
    build_safe_llm_log_payload,
    find_forbidden_llm_log_fields,
)
from app.core.trace import reset_trace_id, set_trace_id


def test_build_safe_llm_success_log_payload_keeps_operational_metadata() -> None:
    payload = build_safe_llm_log_payload(
        operation="chat",
        outcome="success",
        provider="openai-compatible",
        model="qwen3.7-plus",
        elapsed_ms=123.456,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        trace_id="trace-llm-001",
    )

    assert payload == {
        "app.trace_id": "trace-llm-001",
        "llm.operation": "chat",
        "llm.outcome": "success",
        "llm.provider": "openai-compatible",
        "llm.model": "qwen3.7-plus",
        "llm.elapsed_ms": 123.46,
        "llm.prompt_tokens": 100,
        "llm.completion_tokens": 20,
        "llm.total_tokens": 120,
    }


def test_build_safe_llm_failure_log_payload_keeps_error_summary_only() -> None:
    payload = build_safe_llm_log_payload(
        operation="stream_chat",
        outcome="failure",
        provider="openai-compatible",
        model="qwen3.7-plus",
        elapsed_ms=3000,
        error_code="LLM_TIMEOUT",
        status_code=504,
        trace_id="trace-llm-timeout",
    )

    assert payload["llm.outcome"] == "failure"
    assert payload["llm.error_code"] == "LLM_TIMEOUT"
    assert payload["http.status_code"] == 504
    assert payload["llm.elapsed_ms"] == 3000


def test_safe_llm_log_payload_omits_prompt_messages_keys_and_responses() -> None:
    payload = build_safe_llm_log_payload(
        operation="tool_decision",
        outcome="success",
        provider="openai-compatible",
        model="qwen3.7-plus",
        trace_id="trace-llm-sensitive",
        extra_fields={
            "prompt": "full prompt should not be logged",
            "messages": [{"role": "user", "content": "private message"}],
            "user_message": "my phone is 13800000000",
            "Authorization": "Bearer secret-token",
            "api_key": "sk-test-secret",
            "raw_response": {"choices": ["full response"]},
            "final_answer": "private final answer",
            "custom.retry_count": 1,
            "custom.fallback_used": True,
            "custom.route": "/tool-chat",
        },
    )

    assert payload["custom.retry_count"] == 1
    assert payload["custom.fallback_used"] is True
    assert payload["custom.route"] == "/tool-chat"
    assert "prompt" not in payload
    assert "messages" not in payload
    assert "user_message" not in payload
    assert "authorization" not in payload
    assert "api_key" not in payload
    assert "raw_response" not in payload
    assert "final_answer" not in payload


def test_extra_fields_cannot_override_protected_llm_log_fields() -> None:
    payload = build_safe_llm_log_payload(
        operation="chat",
        outcome="success",
        provider="provider-a",
        model="model-a",
        trace_id="trace-protected",
        extra_fields={
            "llm.model": "wrong-model",
            "llm.provider": "wrong-provider",
            "llm.outcome": "failure",
            "app.trace_id": "wrong-trace",
            "safe.source": "unit-test",
        },
    )

    assert payload["llm.model"] == "model-a"
    assert payload["llm.provider"] == "provider-a"
    assert payload["llm.outcome"] == "success"
    assert payload["app.trace_id"] == "trace-protected"
    assert payload["safe.source"] == "unit-test"


def test_safe_llm_log_payload_ignores_invalid_scalar_values() -> None:
    payload = build_safe_llm_log_payload(
        operation="rag_final_answer",
        outcome="success",
        provider="openai-compatible",
        model="qwen3.7-plus",
        elapsed_ms=-1,
        prompt_tokens=-10,
        completion_tokens=True,
        total_tokens=30,
        extra_fields={
            "empty": "   ",
            "complex": {"nested": True},
            "safe.count": 2,
        },
    )

    assert "llm.elapsed_ms" not in payload
    assert "llm.prompt_tokens" not in payload
    assert "llm.completion_tokens" not in payload
    assert payload["llm.total_tokens"] == 30
    assert "empty" not in payload
    assert "complex" not in payload
    assert payload["safe.count"] == 2


def test_find_forbidden_llm_log_fields_reports_sensitive_keys() -> None:
    forbidden = find_forbidden_llm_log_fields(
        {
            "Prompt": "private prompt",
            "messages": [],
            "llm.model": "qwen3.7-plus",
            "api_key": "sk-test-secret",
            "safe_field": "ok",
        }
    )

    assert forbidden == ["api_key", "messages", "prompt"]


def test_safe_llm_log_payload_reuses_current_trace_id() -> None:
    token = set_trace_id("current-llm-trace")
    try:
        payload = build_safe_llm_log_payload(
            operation="chat",
            outcome="success",
            provider="openai-compatible",
            model="qwen3.7-plus",
        )
    finally:
        reset_trace_id(token)

    assert payload["app.trace_id"] == "current-llm-trace"
