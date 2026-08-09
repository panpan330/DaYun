from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Literal


MILLION_TOKENS = 1_000_000
TokenCostStatus = Literal["estimated", "missing_pricing", "incomplete_usage"]
SafeTokenCostLogValue = str | int | float | bool


@dataclass(frozen=True)
class TokenBudget:
    estimated_input_tokens: int
    max_output_tokens: int

    @property
    def total_reserved_tokens(self) -> int:
        return self.estimated_input_tokens + self.max_output_tokens


@dataclass(frozen=True)
class TokenUsageSnapshot:
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
class TokenPricing:
    input_cost_per_million_tokens: float
    output_cost_per_million_tokens: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        _validate_token_price(self.input_cost_per_million_tokens)
        _validate_token_price(self.output_cost_per_million_tokens)
        if not self.currency.strip():
            raise ValueError("currency must not be blank")


@dataclass(frozen=True)
class TokenCostEstimate:
    status: TokenCostStatus
    currency: str | None = None
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None

    @property
    def is_estimated(self) -> bool:
        return self.status == "estimated"


@dataclass(frozen=True)
class TokenCostRecord:
    provider: str
    model: str
    operation: str
    usage: TokenUsageSnapshot
    cost: TokenCostEstimate

    def to_log_fields(self) -> dict[str, SafeTokenCostLogValue]:
        fields: dict[str, SafeTokenCostLogValue] = {
            "llm.cost_status": self.cost.status,
        }
        _add_optional_log_field(fields, "llm.cost_currency", self.cost.currency)
        _add_optional_log_field(fields, "llm.estimated_input_cost", self.cost.input_cost)
        _add_optional_log_field(
            fields,
            "llm.estimated_output_cost",
            self.cost.output_cost,
        )
        _add_optional_log_field(fields, "llm.estimated_total_cost", self.cost.total_cost)
        return fields


def estimate_text_tokens_roughly(text: str) -> int:
    stripped_text = text.strip()
    if not stripped_text:
        return 0

    ascii_count = sum(1 for character in stripped_text if character.isascii())
    non_ascii_count = len(stripped_text) - ascii_count

    return max(1, ceil(ascii_count / 4) + non_ascii_count)


def build_token_budget(text: str, max_output_tokens: int) -> TokenBudget:
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be greater than 0")

    return TokenBudget(
        estimated_input_tokens=estimate_text_tokens_roughly(text),
        max_output_tokens=max_output_tokens,
    )


def normalize_token_usage(usage: Any) -> TokenUsageSnapshot:
    if usage is None:
        return TokenUsageSnapshot()
    if isinstance(usage, TokenUsageSnapshot):
        return usage

    return TokenUsageSnapshot(
        prompt_tokens=_read_token_count(usage, "prompt_tokens", "input_tokens"),
        completion_tokens=_read_token_count(
            usage,
            "completion_tokens",
            "output_tokens",
        ),
        total_tokens=_read_token_count(usage, "total_tokens"),
    )


def estimate_token_cost(
    usage: Any,
    pricing: TokenPricing | None,
) -> TokenCostEstimate:
    usage_snapshot = normalize_token_usage(usage)
    if pricing is None:
        return TokenCostEstimate(status="missing_pricing")
    if (
        usage_snapshot.prompt_tokens is None
        or usage_snapshot.completion_tokens is None
    ):
        return TokenCostEstimate(
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
    return TokenCostEstimate(
        status="estimated",
        currency=pricing.currency.strip(),
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        total_cost=round(input_cost + output_cost, 8),
    )


def build_token_cost_record(
    usage: Any,
    *,
    provider: str,
    model: str,
    operation: str,
    pricing: TokenPricing | None,
) -> TokenCostRecord:
    usage_snapshot = normalize_token_usage(usage)
    return TokenCostRecord(
        provider=provider.strip() or "unknown",
        model=model.strip() or "unknown",
        operation=operation.strip() or "unknown",
        usage=usage_snapshot,
        cost=estimate_token_cost(usage_snapshot, pricing),
    )


def _read_token_count(usage: Any, *field_names: str) -> int | None:
    for field_name in field_names:
        if isinstance(usage, Mapping):
            value = usage.get(field_name)
        else:
            value = getattr(usage, field_name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _validate_token_price(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("token price must be a number")
    if not isfinite(value) or value < 0:
        raise ValueError("token price must be finite and non-negative")


def _add_optional_log_field(
    fields: dict[str, SafeTokenCostLogValue],
    key: str,
    value: object,
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        fields[key] = value
    elif isinstance(value, int):
        fields[key] = value
    elif isinstance(value, float) and isfinite(value):
        fields[key] = value
    elif isinstance(value, str) and value.strip():
        fields[key] = value.strip()
