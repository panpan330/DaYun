import pytest

from app.core.token_usage import (
    TokenPricing,
    TokenUsageSnapshot,
    build_token_budget,
    build_token_cost_record,
    estimate_text_tokens_roughly,
    estimate_token_cost,
    normalize_token_usage,
)


class UsageObject:
    def __init__(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def test_estimate_text_tokens_roughly_returns_zero_for_blank_text() -> None:
    assert estimate_text_tokens_roughly("") == 0
    assert estimate_text_tokens_roughly("   ") == 0


def test_estimate_text_tokens_roughly_estimates_english_text() -> None:
    assert estimate_text_tokens_roughly("abcdefghijkl") == 3


def test_estimate_text_tokens_roughly_counts_non_ascii_more_conservatively() -> None:
    assert estimate_text_tokens_roughly("你好世界") == 4


def test_estimate_text_tokens_roughly_estimates_mixed_text() -> None:
    assert estimate_text_tokens_roughly("hello 你好") == 4


def test_build_token_budget_combines_input_estimate_and_output_limit() -> None:
    budget = build_token_budget("abcdefghijkl", max_output_tokens=100)

    assert budget.estimated_input_tokens == 3
    assert budget.max_output_tokens == 100
    assert budget.total_reserved_tokens == 103


def test_build_token_budget_rejects_invalid_output_limit() -> None:
    with pytest.raises(ValueError, match="max_output_tokens"):
        build_token_budget("hello", max_output_tokens=0)


def test_normalize_token_usage_accepts_dict_object_and_response_api_aliases() -> None:
    dict_usage = normalize_token_usage(
        {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
    )
    object_usage = normalize_token_usage(
        UsageObject(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    alias_usage = normalize_token_usage(
        {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    )

    assert dict_usage == TokenUsageSnapshot(
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=130,
    )
    assert dict_usage.computed_total_tokens == 130
    assert dict_usage.total_matches_split is True
    assert object_usage.prompt_tokens == 10
    assert object_usage.completion_tokens == 5
    assert alias_usage.prompt_tokens == 8
    assert alias_usage.completion_tokens == 4


def test_normalize_token_usage_ignores_invalid_counts() -> None:
    usage = normalize_token_usage(
        {"prompt_tokens": -1, "completion_tokens": True, "total_tokens": "10"}
    )

    assert usage == TokenUsageSnapshot()


def test_estimate_token_cost_uses_per_million_pricing() -> None:
    cost = estimate_token_cost(
        {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        TokenPricing(
            input_cost_per_million_tokens=2.0,
            output_cost_per_million_tokens=6.0,
            currency="USD",
        ),
    )

    assert cost.status == "estimated"
    assert cost.currency == "USD"
    assert cost.input_cost == 0.002
    assert cost.output_cost == 0.003
    assert cost.total_cost == 0.005
    assert cost.is_estimated is True


def test_estimate_token_cost_reports_missing_pricing_or_incomplete_usage() -> None:
    assert (
        estimate_token_cost(
            {"prompt_tokens": 100, "completion_tokens": 20},
            None,
        ).status
        == "missing_pricing"
    )

    incomplete = estimate_token_cost(
        {"total_tokens": 120},
        TokenPricing(
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        ),
    )

    assert incomplete.status == "incomplete_usage"
    assert incomplete.total_cost is None


def test_token_pricing_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenPricing(
            input_cost_per_million_tokens=-1,
            output_cost_per_million_tokens=1,
        )

    with pytest.raises(ValueError, match="finite"):
        TokenPricing(
            input_cost_per_million_tokens=float("inf"),
            output_cost_per_million_tokens=1,
        )

    with pytest.raises(ValueError, match="currency"):
        TokenPricing(
            input_cost_per_million_tokens=1,
            output_cost_per_million_tokens=1,
            currency=" ",
        )


def test_build_token_cost_record_returns_safe_log_fields() -> None:
    record = build_token_cost_record(
        {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        pricing=TokenPricing(
            input_cost_per_million_tokens=2.0,
            output_cost_per_million_tokens=6.0,
            currency="USD",
        ),
    )

    assert record.provider == "dashscope"
    assert record.model == "qwen3.7-plus"
    assert record.operation == "chat"
    assert record.cost.total_cost == 0.005
    assert record.to_log_fields() == {
        "llm.cost_status": "estimated",
        "llm.cost_currency": "USD",
        "llm.estimated_input_cost": 0.002,
        "llm.estimated_output_cost": 0.003,
        "llm.estimated_total_cost": 0.005,
    }
