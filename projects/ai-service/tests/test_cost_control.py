import pytest

from app.core.config import Settings
from app.core.cost_control import (
    build_llm_cost_control_decision,
    estimate_messages_tokens_roughly,
)
from app.core.token_usage import TokenPricing


def test_cost_control_allows_request_within_budget() -> None:
    settings = Settings(_env_file=None)
    messages = [{"role": "user", "content": "解释 FastAPI"}]

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=messages,
        requested_max_output_tokens=256,
    )

    assert decision.action == "allow"
    assert decision.reason == "within_budget"
    assert decision.effective_max_output_tokens == 256
    assert decision.should_block is False


def test_cost_control_caps_output_when_total_budget_can_still_answer() -> None:
    messages = [{"role": "user", "content": "业务流程" * 30}]
    estimated_input_tokens = estimate_messages_tokens_roughly(messages)
    settings = Settings(
        llm_max_total_tokens_per_request=estimated_input_tokens + 200,
        llm_min_output_tokens=128,
        _env_file=None,
    )

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=messages,
        requested_max_output_tokens=1024,
    )

    assert decision.action == "cap_output"
    assert decision.reason == "total_tokens_exceeded"
    assert decision.output_was_capped is True
    assert decision.effective_max_output_tokens == 200
    assert decision.reserved_total_tokens == settings.llm_max_total_tokens_per_request


def test_cost_control_blocks_when_input_budget_is_exceeded() -> None:
    settings = Settings(
        llm_max_input_tokens_per_request=100,
        _env_file=None,
    )
    messages = [{"role": "user", "content": "业务流程" * 150}]

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=messages,
        requested_max_output_tokens=256,
    )

    assert decision.action == "block"
    assert decision.reason == "input_tokens_exceeded"
    assert decision.should_block is True
    assert decision.fallback_allowed is False


def test_cost_control_blocks_when_remaining_output_budget_is_too_small() -> None:
    messages = [{"role": "user", "content": "业务流程" * 80}]
    estimated_input_tokens = estimate_messages_tokens_roughly(messages)
    settings = Settings(
        llm_max_total_tokens_per_request=estimated_input_tokens + 20,
        llm_min_output_tokens=128,
        _env_file=None,
    )

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=messages,
        requested_max_output_tokens=1024,
    )

    assert decision.action == "block"
    assert decision.reason == "total_tokens_exceeded"
    assert decision.effective_max_output_tokens == 20


def test_cost_control_blocks_when_estimated_cost_is_too_high() -> None:
    settings = Settings(
        llm_max_estimated_cost_per_request=0.000001,
        _env_file=None,
    )

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=[{"role": "user", "content": "解释 FastAPI"}],
        requested_max_output_tokens=256,
        pricing=TokenPricing(
            input_cost_per_million_tokens=100.0,
            output_cost_per_million_tokens=100.0,
            currency="USD",
        ),
    )

    assert decision.action == "block"
    assert decision.reason == "estimated_cost_exceeded"
    assert decision.estimated_cost.total_cost is not None


def test_cost_control_disables_fallback_above_reserved_token_threshold() -> None:
    settings = Settings(
        llm_disable_fallback_above_total_tokens=100,
        _env_file=None,
    )

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=[{"role": "user", "content": "解释 FastAPI"}],
        requested_max_output_tokens=256,
    )

    assert decision.action == "allow"
    assert decision.fallback_allowed is False


def test_cost_control_can_be_disabled() -> None:
    settings = Settings(
        llm_enable_cost_control=False,
        llm_disable_fallback_above_total_tokens=100,
        _env_file=None,
    )

    decision = build_llm_cost_control_decision(
        settings,
        serialized_messages=[{"role": "user", "content": "解释 FastAPI"}],
        requested_max_output_tokens=9999,
    )

    assert decision.action == "allow"
    assert decision.reason == "disabled"
    assert decision.effective_max_output_tokens == 9999


def test_cost_control_rejects_invalid_requested_output_tokens() -> None:
    with pytest.raises(ValueError, match="requested_max_output_tokens"):
        build_llm_cost_control_decision(
            Settings(_env_file=None),
            serialized_messages=[],
            requested_max_output_tokens=0,
        )


def test_cost_control_log_fields_do_not_include_message_content() -> None:
    decision = build_llm_cost_control_decision(
        Settings(_env_file=None),
        serialized_messages=[{"role": "user", "content": "用户真实问题"}],
        requested_max_output_tokens=256,
    )

    serialized = str(decision.to_log_fields())

    assert "用户真实问题" not in serialized
    assert "llm.cost_control_action" in serialized
