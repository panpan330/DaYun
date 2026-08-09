from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ScoreDirection = Literal["higher_is_better", "lower_is_better"]
ThresholdOperator = Literal[">=", "<="]
ScoreValueKind = Literal[
    "similarity",
    "distance",
    "match_score",
    "weighted_score",
    "rerank_score",
    "unknown",
]


class RetrievalScoreMeaning(BaseModel):
    backend: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    raw_score_name: str = Field(min_length=1)
    value_kind: ScoreValueKind
    direction: ScoreDirection
    threshold_operator: ThresholdOperator
    range_hint: str = Field(min_length=1)
    can_compare_across_backends: bool = False
    can_compare_across_embedding_models: bool = False
    explanation: str = Field(min_length=1)
    threshold_note: str = Field(min_length=1)

    @field_validator("backend", "metric", "raw_score_name", mode="before")
    @classmethod
    def strip_text_fields(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


def describe_qdrant_score(distance: str) -> RetrievalScoreMeaning:
    metric = _normalize_metric(distance)
    return _build_vector_score_meaning(
        backend="qdrant",
        metric=metric,
        raw_score_name="score",
    )


def describe_milvus_score(metric_type: str) -> RetrievalScoreMeaning:
    metric = _normalize_metric(metric_type)
    return _build_vector_score_meaning(
        backend="milvus",
        metric=metric,
        raw_score_name="distance",
    )


def describe_keyword_score() -> RetrievalScoreMeaning:
    return RetrievalScoreMeaning(
        backend="local_keyword",
        metric="keyword_match",
        raw_score_name="score",
        value_kind="match_score",
        direction="higher_is_better",
        threshold_operator=">=",
        range_hint="0 to 1 in the current learning implementation",
        explanation=(
            "Keyword score is produced by local term matching, not by an embedding "
            "model or vector database."
        ),
        threshold_note="A larger keyword score means more or stronger query terms matched.",
    )


def describe_hybrid_score() -> RetrievalScoreMeaning:
    return RetrievalScoreMeaning(
        backend="local_hybrid",
        metric="weighted_fusion",
        raw_score_name="hybrid_score",
        value_kind="weighted_score",
        direction="higher_is_better",
        threshold_operator=">=",
        range_hint="0 to vector_weight + keyword_weight in the current implementation",
        explanation=(
            "Hybrid score is a local weighted fusion of normalized vector and "
            "keyword scores."
        ),
        threshold_note=(
            "Do not compare this value with raw vector-store scores; it belongs to "
            "the local fusion formula."
        ),
    )


def describe_rerank_score(*, backend: str = "rerank_model") -> RetrievalScoreMeaning:
    return RetrievalScoreMeaning(
        backend=backend,
        metric="model_relevance",
        raw_score_name="rerank_score",
        value_kind="rerank_score",
        direction="higher_is_better",
        threshold_operator=">=",
        range_hint="provider-specific, often 0 to 1",
        explanation=(
            "Rerank score is produced after retrieval by scoring a query against "
            "candidate documents."
        ),
        threshold_note=(
            "Do not compare rerank scores directly with vector, keyword, or hybrid "
            "retrieval scores."
        ),
    )


def is_score_passing_threshold(
    score: float,
    threshold: float | None,
    meaning: RetrievalScoreMeaning,
) -> bool:
    _validate_score_number(score, field_name="score")
    if threshold is None:
        return True
    _validate_score_number(threshold, field_name="threshold")

    if meaning.direction == "lower_is_better":
        return score <= threshold
    return score >= threshold


def filter_scores_by_threshold(
    scores: Sequence[float],
    *,
    threshold: float | None,
    meaning: RetrievalScoreMeaning,
) -> list[float]:
    return [
        score
        for score in scores
        if is_score_passing_threshold(score, threshold, meaning)
    ]


def sort_scores_by_relevance(
    scores: Iterable[float],
    meaning: RetrievalScoreMeaning,
) -> list[float]:
    return sorted(
        scores,
        reverse=meaning.direction == "higher_is_better",
    )


def format_score_meaning_for_debug(
    score: float,
    meaning: RetrievalScoreMeaning,
    *,
    threshold: float | None = None,
) -> str:
    _validate_score_number(score, field_name="score")
    threshold_text = "none" if threshold is None else f"{threshold:.4f}"
    return (
        f"{meaning.backend}/{meaning.metric} "
        f"{meaning.raw_score_name}={score:.4f} "
        f"direction={meaning.direction} "
        f"threshold_operator={meaning.threshold_operator} "
        f"threshold={threshold_text}"
    )


def _build_vector_score_meaning(
    *,
    backend: str,
    metric: str,
    raw_score_name: str,
) -> RetrievalScoreMeaning:
    if metric in {"cosine", "dot", "ip"}:
        return RetrievalScoreMeaning(
            backend=backend,
            metric=metric,
            raw_score_name=raw_score_name,
            value_kind="similarity",
            direction="higher_is_better",
            threshold_operator=">=",
            range_hint=_range_hint_for_metric(metric),
            explanation=(
                "This metric returns a similarity-style value in this project: "
                "larger values should be treated as more relevant."
            ),
            threshold_note=(
                "A threshold keeps results whose raw score is greater than or "
                "equal to the threshold."
            ),
        )

    if metric in {"l2", "euclid", "manhattan"}:
        return RetrievalScoreMeaning(
            backend=backend,
            metric=metric,
            raw_score_name=raw_score_name,
            value_kind="distance",
            direction="lower_is_better",
            threshold_operator="<=",
            range_hint=_range_hint_for_metric(metric),
            explanation=(
                "This metric returns a distance-style value in this project: "
                "smaller values should be treated as more relevant."
            ),
            threshold_note=(
                "A threshold keeps results whose raw distance is less than or "
                "equal to the threshold."
            ),
        )

    raise ValueError("unsupported retrieval score metric")


def _normalize_metric(metric: str) -> str:
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-blank string")
    normalized = metric.strip().lower()
    aliases = {
        "cos": "cosine",
        "cosine": "cosine",
        "dot": "dot",
        "inner_product": "ip",
        "ip": "ip",
        "l2": "l2",
        "euclid": "l2",
        "euclidean": "l2",
        "manhattan": "manhattan",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        raise ValueError("unsupported retrieval score metric")
    return mapped


def _range_hint_for_metric(metric: str) -> str:
    if metric == "cosine":
        return "-1 to 1 in the usual cosine definition"
    if metric in {"dot", "ip"}:
        return "depends on vector magnitude and normalization"
    if metric in {"l2", "euclid", "manhattan"}:
        return "0 to infinity"
    return "unknown"


def _validate_score_number(value: float, *, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
