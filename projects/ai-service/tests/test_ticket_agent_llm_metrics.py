from dataclasses import dataclass

import pytest

from app.agents.llm_metrics import (
    APP_LLM_ERRORS_METRIC_NAME,
    APP_LLM_ESTIMATED_COST_METRIC_NAME,
    APP_LLM_REQUESTS_METRIC_NAME,
    GEN_AI_OPERATION_DURATION_METRIC_NAME,
    GEN_AI_TOKEN_USAGE_METRIC_NAME,
    LLMTokenPricing,
    build_llm_call_metrics,
    build_llm_metric_attributes,
    build_llm_metric_specs,
    estimate_llm_call_cost,
    normalize_llm_token_usage,
)


@dataclass(frozen=True)
class UsageObject:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def test_normalize_llm_token_usage_accepts_dict_and_object_shapes() -> None:
    dict_usage = normalize_llm_token_usage(
        {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
    )
    object_usage = normalize_llm_token_usage(
        UsageObject(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )

    assert dict_usage.prompt_tokens == 100
    assert dict_usage.completion_tokens == 30
    assert dict_usage.total_tokens == 130
    assert dict_usage.computed_total_tokens == 130
    assert dict_usage.total_matches_split is True
    assert object_usage.prompt_tokens == 10
    assert object_usage.completion_tokens == 5
    assert object_usage.total_tokens == 15


def test_normalize_llm_token_usage_ignores_invalid_counts() -> None:
    usage = normalize_llm_token_usage(
        {"prompt_tokens": -1, "completion_tokens": True, "total_tokens": "10"}
    )

    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_estimate_llm_call_cost_uses_configured_per_million_prices() -> None:
    cost = estimate_llm_call_cost(
        {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        LLMTokenPricing(
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


def test_estimate_llm_call_cost_reports_missing_pricing_or_incomplete_usage() -> None:
    assert (
        estimate_llm_call_cost(
            {"prompt_tokens": 100, "completion_tokens": 20},
            None,
        ).status
        == "missing_pricing"
    )

    incomplete = estimate_llm_call_cost(
        {"total_tokens": 120},
        LLMTokenPricing(
            input_cost_per_million_tokens=1.0,
            output_cost_per_million_tokens=2.0,
        ),
    )

    assert incomplete.status == "incomplete_usage"
    assert incomplete.total_cost is None


def test_pricing_rejects_negative_or_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LLMTokenPricing(
            input_cost_per_million_tokens=-1,
            output_cost_per_million_tokens=1,
        )

    with pytest.raises(ValueError, match="finite"):
        LLMTokenPricing(
            input_cost_per_million_tokens=float("inf"),
            output_cost_per_million_tokens=1,
        )


def test_build_llm_call_metrics_emits_request_duration_token_and_cost_metrics() -> None:
    metrics = build_llm_call_metrics(
        {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        status="ok",
        llm_task="ticket_intent_classification",
        prompt_name="ticket_intent_classification",
        prompt_version="v1",
        elapsed_ms=1234.567,
        pricing=LLMTokenPricing(
            input_cost_per_million_tokens=2.0,
            output_cost_per_million_tokens=6.0,
            currency="USD",
        ),
    )

    by_name = {}
    for metric in metrics:
        by_name.setdefault(metric.name, []).append(metric)

    assert [metric.name for metric in metrics] == [
        APP_LLM_REQUESTS_METRIC_NAME,
        GEN_AI_OPERATION_DURATION_METRIC_NAME,
        GEN_AI_TOKEN_USAGE_METRIC_NAME,
        GEN_AI_TOKEN_USAGE_METRIC_NAME,
        APP_LLM_ESTIMATED_COST_METRIC_NAME,
    ]
    request_metric = by_name[APP_LLM_REQUESTS_METRIC_NAME][0]
    duration_metric = by_name[GEN_AI_OPERATION_DURATION_METRIC_NAME][0]
    token_metrics = by_name[GEN_AI_TOKEN_USAGE_METRIC_NAME]
    cost_metric = by_name[APP_LLM_ESTIMATED_COST_METRIC_NAME][0]

    assert request_metric.kind == "counter"
    assert request_metric.value == 1
    assert request_metric.attributes["gen_ai.provider.name"] == "dashscope"
    assert request_metric.attributes["gen_ai.request.model"] == "qwen3.7-plus"
    assert request_metric.attributes["gen_ai.operation.name"] == "chat"
    assert request_metric.attributes["app.llm.task"] == (
        "ticket_intent_classification"
    )
    assert duration_metric.kind == "histogram"
    assert duration_metric.value == 1.234567
    assert duration_metric.unit == "s"
    assert {
        metric.attributes["gen_ai.token.type"]: metric.value
        for metric in token_metrics
    } == {"input": 1000, "output": 500}
    assert cost_metric.kind == "counter"
    assert cost_metric.unit == "USD"
    assert cost_metric.value == 0.005
    assert cost_metric.attributes["currency"] == "USD"


def test_error_metrics_include_error_type_but_do_not_require_token_usage() -> None:
    metrics = build_llm_call_metrics(
        None,
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        status="error",
        error_code="LLM_TIMEOUT",
        elapsed_ms=3000,
    )

    assert [metric.name for metric in metrics] == [
        APP_LLM_REQUESTS_METRIC_NAME,
        APP_LLM_ERRORS_METRIC_NAME,
        GEN_AI_OPERATION_DURATION_METRIC_NAME,
    ]
    for metric in metrics:
        assert metric.attributes["status"] == "error"
        assert metric.attributes["error.type"] == "LLM_TIMEOUT"
    assert metrics[-1].value == 3.0


def test_metric_attributes_filter_high_cardinality_and_sensitive_context() -> None:
    attributes = build_llm_metric_attributes(
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        status="ok",
        prompt_name="ticket_intent_classification",
        extra_attributes={
            "trace_id": "8b0e715c76c8423e9dc95b6c8db8409a",
            "span_id": "5fb397be34d26b51",
            "thread_id": "ticket-thread-001",
            "user_message": "包含手机号的用户原文",
            "prompt": "完整提示词",
            "business_domain": "ticket",
        },
    )

    assert attributes["business_domain"] == "ticket"
    assert "trace_id" not in attributes
    assert "span_id" not in attributes
    assert "thread_id" not in attributes
    assert "user_message" not in attributes
    assert "prompt" not in attributes


def test_missing_pricing_does_not_emit_estimated_cost_metric() -> None:
    metrics = build_llm_call_metrics(
        {"prompt_tokens": 1000, "completion_tokens": 500},
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        elapsed_ms=100,
        pricing=None,
    )

    assert APP_LLM_ESTIMATED_COST_METRIC_NAME not in [
        metric.name for metric in metrics
    ]


def test_missing_split_usage_does_not_emit_token_or_cost_metrics() -> None:
    metrics = build_llm_call_metrics(
        {"total_tokens": 1500},
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        pricing=LLMTokenPricing(
            input_cost_per_million_tokens=2.0,
            output_cost_per_million_tokens=6.0,
        ),
    )

    assert [metric.name for metric in metrics] == [APP_LLM_REQUESTS_METRIC_NAME]


def test_invalid_duration_is_not_recorded_as_histogram() -> None:
    metrics = build_llm_call_metrics(
        {"prompt_tokens": 1, "completion_tokens": 1},
        provider="dashscope",
        model="qwen3.7-plus",
        operation="chat",
        elapsed_ms=-1,
    )

    assert GEN_AI_OPERATION_DURATION_METRIC_NAME not in [
        metric.name for metric in metrics
    ]


def test_llm_metric_specs_document_instrument_choices_and_required_attributes() -> None:
    specs = {spec.name: spec for spec in build_llm_metric_specs()}

    assert specs[GEN_AI_OPERATION_DURATION_METRIC_NAME].kind == "histogram"
    assert specs[GEN_AI_OPERATION_DURATION_METRIC_NAME].unit == "s"
    assert specs[GEN_AI_TOKEN_USAGE_METRIC_NAME].kind == "histogram"
    assert specs[GEN_AI_TOKEN_USAGE_METRIC_NAME].unit == "{token}"
    assert "gen_ai.token.type" in (
        specs[GEN_AI_TOKEN_USAGE_METRIC_NAME].required_attributes
    )
    assert specs[APP_LLM_REQUESTS_METRIC_NAME].kind == "counter"
    assert specs[APP_LLM_ERRORS_METRIC_NAME].kind == "counter"
    assert specs[APP_LLM_ESTIMATED_COST_METRIC_NAME].kind == "counter"
