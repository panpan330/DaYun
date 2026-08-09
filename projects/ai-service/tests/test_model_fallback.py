from app.core.config import Settings
from app.core.model_fallback import (
    build_llm_fallback_decision,
    parse_fallback_error_codes,
)
from app.core.model_routing import route_llm_model


def test_fallback_decision_attempts_retryable_error_with_distinct_model() -> None:
    settings = Settings(
        llm_model="qwen-balanced",
        llm_fast_model="qwen-fast",
        llm_balanced_model="qwen-balanced",
        llm_fallback_tier="balanced",
        _env_file=None,
    )
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我摘要这段文字",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="LLM_TIMEOUT",
    )

    assert decision.should_attempt is True
    assert decision.reason == "retryable_error"
    assert decision.primary_model == "qwen-fast"
    assert decision.fallback_route is not None
    assert decision.fallback_route.model == "qwen-balanced"
    assert decision.fallback_route.tier == "balanced"


def test_fallback_decision_skips_non_retryable_error() -> None:
    settings = Settings(
        llm_model="qwen-balanced",
        llm_fast_model="qwen-fast",
        _env_file=None,
    )
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我摘要这段文字",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="LLM_AUTHENTICATION_FAILED",
    )

    assert decision.should_attempt is False
    assert decision.reason == "non_retryable_error"
    assert decision.fallback_route is None


def test_fallback_decision_skips_when_disabled() -> None:
    settings = Settings(
        llm_model="qwen-balanced",
        llm_fast_model="qwen-fast",
        llm_enable_fallback=False,
        _env_file=None,
    )
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我摘要这段文字",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="LLM_TIMEOUT",
    )

    assert decision.should_attempt is False
    assert decision.reason == "disabled"


def test_fallback_decision_skips_same_model() -> None:
    settings = Settings(llm_model="qwen-default", _env_file=None)
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="普通问题",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="LLM_TIMEOUT",
    )

    assert decision.should_attempt is False
    assert decision.reason == "same_model"
    assert decision.fallback_route is not None
    assert decision.fallback_route.model == "qwen-default"


def test_fallback_decision_uses_explicit_fallback_model() -> None:
    settings = Settings(
        llm_model="qwen-balanced",
        llm_fast_model="qwen-fast",
        llm_fallback_model="qwen-backup",
        _env_file=None,
    )
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我摘要这段文字",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="LLM_PROVIDER_ERROR",
    )

    assert decision.should_attempt is True
    assert decision.fallback_route is not None
    assert decision.fallback_route.model == "qwen-backup"


def test_fallback_log_fields_do_not_include_prompt_text() -> None:
    settings = Settings(
        llm_model="qwen-balanced",
        llm_fast_model="qwen-fast",
        _env_file=None,
    )
    primary_route = route_llm_model(
        settings,
        operation="chat",
        input_text="用户真实问题",
    )

    decision = build_llm_fallback_decision(
        settings,
        primary_route=primary_route,
        error_code="llm_timeout",
    )
    fields = decision.to_log_fields()
    serialized = str(fields)

    assert fields["llm.primary_error_code"] == "LLM_TIMEOUT"
    assert "用户真实问题" not in serialized


def test_parse_fallback_error_codes_splits_and_normalizes() -> None:
    assert parse_fallback_error_codes(" llm_timeout，LLM_RATE_LIMITED;llm_timeout ") == (
        frozenset({"LLM_TIMEOUT", "LLM_RATE_LIMITED"})
    )
