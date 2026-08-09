import pytest

from app.rag.score_interpretation import (
    describe_hybrid_score,
    describe_keyword_score,
    describe_milvus_score,
    describe_qdrant_score,
    describe_rerank_score,
    filter_scores_by_threshold,
    format_score_meaning_for_debug,
    is_score_passing_threshold,
    sort_scores_by_relevance,
)


def test_describe_qdrant_cosine_score_as_higher_is_better() -> None:
    meaning = describe_qdrant_score("Cosine")

    assert meaning.backend == "qdrant"
    assert meaning.metric == "cosine"
    assert meaning.raw_score_name == "score"
    assert meaning.value_kind == "similarity"
    assert meaning.direction == "higher_is_better"
    assert meaning.threshold_operator == ">="
    assert meaning.can_compare_across_backends is False
    assert meaning.can_compare_across_embedding_models is False


def test_describe_qdrant_euclid_score_as_lower_is_better() -> None:
    meaning = describe_qdrant_score("Euclid")

    assert meaning.metric == "l2"
    assert meaning.value_kind == "distance"
    assert meaning.direction == "lower_is_better"
    assert meaning.threshold_operator == "<="
    assert meaning.range_hint == "0 to infinity"


def test_describe_milvus_metric_aliases() -> None:
    assert describe_milvus_score("COSINE").direction == "higher_is_better"
    assert describe_milvus_score("IP").direction == "higher_is_better"
    assert describe_milvus_score("L2").direction == "lower_is_better"


def test_score_threshold_uses_metric_direction() -> None:
    cosine = describe_milvus_score("COSINE")
    l2 = describe_milvus_score("L2")

    assert is_score_passing_threshold(0.83, 0.8, cosine) is True
    assert is_score_passing_threshold(0.72, 0.8, cosine) is False
    assert is_score_passing_threshold(0.35, 0.5, l2) is True
    assert is_score_passing_threshold(0.91, 0.5, l2) is False


def test_filter_scores_by_threshold_keeps_relevant_values() -> None:
    cosine = describe_qdrant_score("Cosine")
    l2 = describe_qdrant_score("Euclid")

    assert filter_scores_by_threshold(
        [0.9, 0.7, 0.81],
        threshold=0.8,
        meaning=cosine,
    ) == [0.9, 0.81]
    assert filter_scores_by_threshold(
        [0.2, 0.7, 0.4],
        threshold=0.5,
        meaning=l2,
    ) == [0.2, 0.4]


def test_sort_scores_by_relevance_respects_direction() -> None:
    cosine = describe_qdrant_score("Cosine")
    l2 = describe_qdrant_score("Euclid")

    assert sort_scores_by_relevance([0.8, 0.95, 0.6], cosine) == [0.95, 0.8, 0.6]
    assert sort_scores_by_relevance([0.8, 0.2, 0.6], l2) == [0.2, 0.6, 0.8]


def test_describe_local_keyword_and_hybrid_scores() -> None:
    keyword = describe_keyword_score()
    hybrid = describe_hybrid_score()

    assert keyword.backend == "local_keyword"
    assert keyword.value_kind == "match_score"
    assert keyword.direction == "higher_is_better"
    assert hybrid.backend == "local_hybrid"
    assert hybrid.raw_score_name == "hybrid_score"
    assert hybrid.value_kind == "weighted_score"


def test_describe_rerank_score_as_post_retrieval_model_score() -> None:
    meaning = describe_rerank_score(backend="http_reranker")

    assert meaning.backend == "http_reranker"
    assert meaning.raw_score_name == "rerank_score"
    assert meaning.value_kind == "rerank_score"
    assert meaning.direction == "higher_is_better"
    assert meaning.can_compare_across_backends is False


def test_format_score_meaning_for_debug() -> None:
    meaning = describe_milvus_score("L2")

    line = format_score_meaning_for_debug(0.34567, meaning, threshold=0.5)

    assert line == (
        "milvus/l2 distance=0.3457 direction=lower_is_better "
        "threshold_operator=<= threshold=0.5000"
    )


def test_score_interpretation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="metric"):
        describe_qdrant_score("   ")

    with pytest.raises(ValueError, match="unsupported"):
        describe_milvus_score("JACCARD")

    with pytest.raises(ValueError, match="score"):
        is_score_passing_threshold(True, 0.5, describe_qdrant_score("Cosine"))
