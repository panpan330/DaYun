from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from app.agents.observability_signals import HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS


LLMMetricInstrumentKind = Literal["counter", "histogram", "gauge", "up_down_counter"]
LLMTokenType = Literal["input", "output"]
LLMCostEstimationStatus = Literal[
    "estimated",
    "missing_pricing",
    "incomplete_usage",
]
LLMCallStatus = Literal["ok", "error"]

GEN_AI_OPERATION_DURATION_METRIC_NAME = "gen_ai.client.operation.duration"
GEN_AI_TOKEN_USAGE_METRIC_NAME = "gen_ai.client.token.usage"
APP_LLM_REQUESTS_METRIC_NAME = "app.llm.client.requests"
APP_LLM_ERRORS_METRIC_NAME = "app.llm.client.errors"
APP_LLM_ESTIMATED_COST_METRIC_NAME = "app.llm.client.estimated_cost"

MILLION_TOKENS = 1_000_000

LLM_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS = frozenset(
    {
        *HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS,
        "trace_id",
        "span_id",
        "thread_id",
        "session_id",
        "actor_id",
        "user_id",
        "conversation_id",
        "gen_ai.conversation.id",
        "prompt",
        "messages",
        "user_message",
        "normalized_message",
        "final_answer",
        "raw_response",
        "raw_completion",
    }
)


@dataclass(frozen=True)
class LLMMetricSpec:
    name: str
    kind: LLMMetricInstrumentKind
    unit: str
    description: str
    required_attributes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMMetricMeasurement:
    name: str
    kind: LLMMetricInstrumentKind
    value: int | float
    unit: str
    attributes: dict[str, str | int | float | bool]
    description: str


@dataclass(frozen=True)
class LLMTokenUsageSnapshot:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def has_split_usage(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def computed_total_tokens(self) -> int | None:
        if self.has_split_usage:
            return int(self.prompt_tokens or 0) + int(self.completion_tokens or 0)
        return self.total_tokens

    @property
    def total_matches_split(self) -> bool | None:
        if not self.has_split_usage or self.total_tokens is None:
            return None
        return self.total_tokens == self.computed_total_tokens


@dataclass(frozen=True)
class LLMTokenPricing:
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        _validate_price(self.input_cost_per_million_tokens)
        _validate_price(self.output_cost_per_million_tokens)
        if not self.currency.strip():
            raise ValueError("currency must not be blank.")


@dataclass(frozen=True)
class LLMEstimatedCost:
    status: LLMCostEstimationStatus
    currency: str | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None

    @property
    def is_estimated(self) -> bool:
        return self.status == "estimated"


def build_llm_metric_specs() -> list[LLMMetricSpec]:
    return [
        LLMMetricSpec(
            name=APP_LLM_REQUESTS_METRIC_NAME,
            kind="counter",
            unit="{request}",
            description="Number of LLM client requests.",
            required_attributes=(
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.operation.name",
                "status",
            ),
        ),
        LLMMetricSpec(
            name=APP_LLM_ERRORS_METRIC_NAME,
            kind="counter",
            unit="{error}",
            description="Number of failed LLM client requests.",
            required_attributes=(
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.operation.name",
                "error.type",
            ),
        ),
        LLMMetricSpec(
            name=GEN_AI_OPERATION_DURATION_METRIC_NAME,
            kind="histogram",
            unit="s",
            description="Duration of GenAI client operations.",
            required_attributes=(
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.operation.name",
            ),
        ),
        LLMMetricSpec(
            name=GEN_AI_TOKEN_USAGE_METRIC_NAME,
            kind="histogram",
            unit="{token}",
            description="Token usage reported for one GenAI client operation.",
            required_attributes=(
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.operation.name",
                "gen_ai.token.type",
            ),
        ),
        LLMMetricSpec(
            name=APP_LLM_ESTIMATED_COST_METRIC_NAME,
            kind="counter",
            unit="USD",
            description="Estimated LLM call cost based on configured token pricing.",
            required_attributes=(
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.operation.name",
                "currency",
            ),
        ),
    ]


def normalize_llm_token_usage(usage: Any) -> LLMTokenUsageSnapshot:
    if usage is None:
        return LLMTokenUsageSnapshot()

    return LLMTokenUsageSnapshot(
        prompt_tokens=_read_token_count(usage, "prompt_tokens"),
        completion_tokens=_read_token_count(usage, "completion_tokens"),
        total_tokens=_read_token_count(usage, "total_tokens"),
    )


def estimate_llm_call_cost(
    usage: Any,
    pricing: LLMTokenPricing | None,
) -> LLMEstimatedCost:
    usage_snapshot = normalize_llm_token_usage(usage)
    if pricing is None:
        return LLMEstimatedCost(status="missing_pricing")

    if (
        usage_snapshot.prompt_tokens is None
        or usage_snapshot.completion_tokens is None
    ):
        return LLMEstimatedCost(
            status="incomplete_usage",
            currency=pricing.currency,
        )

    input_cost = (
        usage_snapshot.prompt_tokens
        * pricing.input_cost_per_million_tokens
        / MILLION_TOKENS
    )
    output_cost = (
        usage_snapshot.completion_tokens
        * pricing.output_cost_per_million_tokens
        / MILLION_TOKENS
    )
    total_cost = input_cost + output_cost
    return LLMEstimatedCost(
        status="estimated",
        currency=pricing.currency,
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        total_cost=round(total_cost, 8),
    )


def build_llm_metric_attributes(
    *,
    provider: str,
    model: str,
    operation: str,
    status: LLMCallStatus,
    llm_task: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    error_code: str | None = None,
    extra_attributes: Mapping[str, object] | None = None,
) -> dict[str, str | int | float | bool]:
    attributes: dict[str, str | int | float | bool] = {
        "gen_ai.provider.name": _normalize_required_attribute(
            provider,
            fallback="unknown",
        ),
        "gen_ai.request.model": _normalize_required_attribute(
            model,
            fallback="unknown",
        ),
        "gen_ai.operation.name": _normalize_required_attribute(
            operation,
            fallback="unknown",
        ),
        "status": status,
    }
    _add_optional_metric_attribute(attributes, "app.llm.task", llm_task)
    _add_optional_metric_attribute(attributes, "prompt.name", prompt_name)
    _add_optional_metric_attribute(attributes, "prompt.version", prompt_version)
    if status == "error":
        _add_optional_metric_attribute(attributes, "error.type", error_code)

    if extra_attributes is not None:
        for key, value in extra_attributes.items():
            if _is_high_cardinality_metric_attribute(key):
                continue
            _add_optional_metric_attribute(attributes, key, value)

    return {
        key: value
        for key, value in attributes.items()
        if not _is_high_cardinality_metric_attribute(key)
    }


def build_llm_call_metrics(
    usage: Any,
    *,
    provider: str,
    model: str,
    operation: str,
    status: LLMCallStatus = "ok",
    llm_task: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    elapsed_ms: float | None = None,
    error_code: str | None = None,
    pricing: LLMTokenPricing | None = None,
    extra_attributes: Mapping[str, object] | None = None,
) -> list[LLMMetricMeasurement]:
    base_attributes = build_llm_metric_attributes(
        provider=provider,
        model=model,
        operation=operation,
        status=status,
        llm_task=llm_task,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        error_code=error_code,
        extra_attributes=extra_attributes,
    )
    metrics = [
        LLMMetricMeasurement(
            name=APP_LLM_REQUESTS_METRIC_NAME,
            kind="counter",
            value=1,
            unit="{request}",
            attributes=base_attributes,
            description="Number of LLM client requests.",
        )
    ]

    if status == "error":
        metrics.append(
            LLMMetricMeasurement(
                name=APP_LLM_ERRORS_METRIC_NAME,
                kind="counter",
                value=1,
                unit="{error}",
                attributes=base_attributes,
                description="Number of failed LLM client requests.",
            )
        )

    if elapsed_ms is not None and math.isfinite(elapsed_ms) and elapsed_ms >= 0:
        metrics.append(
            LLMMetricMeasurement(
                name=GEN_AI_OPERATION_DURATION_METRIC_NAME,
                kind="histogram",
                value=round(elapsed_ms / 1000, 6),
                unit="s",
                attributes=base_attributes,
                description="Duration of GenAI client operations.",
            )
        )

    usage_snapshot = normalize_llm_token_usage(usage)
    if usage_snapshot.prompt_tokens is not None:
        metrics.append(
            _build_token_usage_metric(
                value=usage_snapshot.prompt_tokens,
                token_type="input",
                base_attributes=base_attributes,
            )
        )
    if usage_snapshot.completion_tokens is not None:
        metrics.append(
            _build_token_usage_metric(
                value=usage_snapshot.completion_tokens,
                token_type="output",
                base_attributes=base_attributes,
            )
        )

    estimated_cost = estimate_llm_call_cost(usage_snapshot, pricing)
    if estimated_cost.is_estimated and estimated_cost.total_cost is not None:
        cost_attributes = dict(base_attributes)
        cost_attributes["currency"] = estimated_cost.currency or "USD"
        metrics.append(
            LLMMetricMeasurement(
                name=APP_LLM_ESTIMATED_COST_METRIC_NAME,
                kind="counter",
                value=estimated_cost.total_cost,
                unit=estimated_cost.currency or "USD",
                attributes=cost_attributes,
                description=(
                    "Estimated LLM call cost based on configured token pricing."
                ),
            )
        )

    return metrics


def _build_token_usage_metric(
    *,
    value: int,
    token_type: LLMTokenType,
    base_attributes: Mapping[str, str | int | float | bool],
) -> LLMMetricMeasurement:
    attributes = dict(base_attributes)
    attributes["gen_ai.token.type"] = token_type
    return LLMMetricMeasurement(
        name=GEN_AI_TOKEN_USAGE_METRIC_NAME,
        kind="histogram",
        value=value,
        unit="{token}",
        attributes=attributes,
        description="Token usage reported for one GenAI client operation.",
    )


def _read_token_count(usage: Any, field_name: str) -> int | None:
    if isinstance(usage, Mapping):
        value = usage.get(field_name)
    else:
        value = getattr(usage, field_name, None)

    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _validate_price(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("token price must be a number.")
    if not math.isfinite(value) or value < 0:
        raise ValueError("token price must be finite and non-negative.")


def _is_high_cardinality_metric_attribute(key: object) -> bool:
    normalized_key = str(key).strip().lower()
    return normalized_key in LLM_HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS


def _add_optional_metric_attribute(
    attributes: dict[str, str | int | float | bool],
    key: object,
    value: object,
) -> None:
    if _is_high_cardinality_metric_attribute(key):
        return
    normalized_value = _normalize_metric_attribute_value(value)
    if normalized_value is not None:
        attributes[str(key)] = normalized_value


def _normalize_required_attribute(value: object, *, fallback: str) -> str:
    normalized = _normalize_metric_attribute_value(value)
    if isinstance(normalized, str):
        return normalized
    return fallback


def _normalize_metric_attribute_value(
    value: object,
) -> str | int | float | bool | None:
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
