import pytest

from app.rag.performance import (
    InMemoryTtlCache,
    RagDegradationMode,
    RagCacheStats,
    RagOperationStage,
    RagOperationStatus,
    assess_operation_timing,
    build_batch_plan,
    build_rag_performance_protection_report,
    build_retrieval_cache_key,
    choose_degradation_decision,
    format_rag_performance_protection_report,
)
from tests.rag_fakes import make_retrieved_chunk


def test_build_retrieval_cache_key_is_stable_and_hides_raw_query() -> None:
    first = build_retrieval_cache_key(
        "  退款多久到账？ ",
        top_k=5,
        score_threshold=0.7,
        permission_group="customer_service",
        embedding_model="text-embedding-demo",
        embedding_dimension=8,
        collection_name="learning_rag_chunks",
    )
    second = build_retrieval_cache_key(
        "退款多久到账？",
        top_k=5,
        score_threshold=0.7,
        permission_group="customer_service",
        embedding_model="text-embedding-demo",
        embedding_dimension=8,
        collection_name="learning_rag_chunks",
    )

    assert first == second
    assert first.value.startswith("rag_retrieval:")
    assert "退款" not in first.value
    assert first.components["query_hash"] != "退款多久到账？"


def test_build_retrieval_cache_key_changes_when_scope_changes() -> None:
    customer_key = build_retrieval_cache_key(
        "退款多久到账？",
        permission_group="customer_service",
        top_k=5,
    )
    internal_key = build_retrieval_cache_key(
        "退款多久到账？",
        permission_group="internal_staff",
        top_k=5,
    )
    top_k_key = build_retrieval_cache_key(
        "退款多久到账？",
        permission_group="customer_service",
        top_k=10,
    )

    assert customer_key.value != internal_key.value
    assert customer_key.value != top_k_key.value


def test_build_retrieval_cache_key_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="query"):
        build_retrieval_cache_key("   ")

    with pytest.raises(ValueError, match="top_k"):
        build_retrieval_cache_key("退款", top_k=True)

    with pytest.raises(ValueError, match="score_threshold"):
        build_retrieval_cache_key("退款", score_threshold=True)


def test_in_memory_ttl_cache_returns_value_before_expiration() -> None:
    now = [100.0]
    cache = InMemoryTtlCache(ttl_seconds=10, clock=lambda: now[0])
    chunk = make_retrieved_chunk(chunk_id="chunk-1")

    cache.set("key-1", [chunk])

    assert cache.get("key-1") == [chunk]
    assert cache.stats().hit_count == 1
    assert cache.stats().miss_count == 0
    assert cache.stats().current_entries == 1


def test_in_memory_ttl_cache_expires_entries() -> None:
    now = [100.0]
    cache = InMemoryTtlCache(ttl_seconds=5, clock=lambda: now[0])

    cache.set("key-1", "value")
    now[0] = 106.0

    assert cache.get("key-1") is None
    stats = cache.stats()
    assert stats.miss_count == 1
    assert stats.evicted_count == 1
    assert stats.current_entries == 0


def test_in_memory_ttl_cache_evicts_oldest_entry_when_full() -> None:
    now = [100.0]
    cache = InMemoryTtlCache(ttl_seconds=60, max_entries=2, clock=lambda: now[0])

    cache.set("key-1", "first")
    now[0] += 1
    cache.set("key-2", "second")
    now[0] += 1
    cache.set("key-3", "third")

    assert cache.get("key-1") is None
    assert cache.get("key-2") == "second"
    assert cache.get("key-3") == "third"
    assert cache.stats().evicted_count == 1


def test_build_batch_plan_splits_items() -> None:
    plan = build_batch_plan(
        ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
        batch_size=2,
    )

    assert plan.item_count == 5
    assert plan.batch_size == 2
    assert plan.batch_count == 3
    assert plan.batches == [
        ["chunk-1", "chunk-2"],
        ["chunk-3", "chunk-4"],
        ["chunk-5"],
    ]


def test_build_batch_plan_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        build_batch_plan(["chunk-1"], batch_size=0)

    with pytest.raises(ValueError, match="batch item"):
        build_batch_plan(["chunk-1", "   "], batch_size=2)


def test_assess_operation_timing_classifies_status() -> None:
    assert assess_operation_timing(
        RagOperationStage.EMBEDDING,
        elapsed_ms=300,
        timeout_seconds=1,
    ).status is RagOperationStatus.OK
    assert assess_operation_timing(
        RagOperationStage.VECTOR_STORE,
        elapsed_ms=850,
        timeout_seconds=1,
    ).status is RagOperationStatus.NEAR_TIMEOUT
    assert assess_operation_timing(
        RagOperationStage.GENERATION,
        elapsed_ms=1000,
        timeout_seconds=1,
    ).status is RagOperationStatus.TIMED_OUT


def test_assess_operation_timing_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        assess_operation_timing(
            RagOperationStage.EMBEDDING,
            elapsed_ms=-1,
            timeout_seconds=1,
        )

    with pytest.raises(ValueError, match="timeout_seconds"):
        assess_operation_timing(
            RagOperationStage.EMBEDDING,
            elapsed_ms=1,
            timeout_seconds=0,
        )


def test_choose_degradation_decision_prefers_cache() -> None:
    decision = choose_degradation_decision(
        RagOperationStage.VECTOR_STORE,
        has_cached_retrieval=True,
        has_safe_chunks=True,
    )

    assert decision.mode is RagDegradationMode.USE_CACHED_RETRIEVAL
    assert decision.should_use_cache is True
    assert decision.should_call_model is True


def test_choose_degradation_decision_uses_safe_fallback_without_cache() -> None:
    decision = choose_degradation_decision(
        RagOperationStage.GENERATION,
        has_cached_retrieval=False,
        has_safe_chunks=True,
    )

    assert decision.mode is RagDegradationMode.RETURN_SAFE_FALLBACK
    assert decision.should_use_cache is False
    assert decision.should_call_model is False


def test_choose_degradation_decision_returns_no_context_without_cache_or_chunks() -> None:
    decision = choose_degradation_decision(
        RagOperationStage.EMBEDDING,
        has_cached_retrieval=False,
        has_safe_chunks=False,
    )

    assert decision.mode is RagDegradationMode.RETURN_NO_CONTEXT
    assert decision.should_use_cache is False
    assert decision.should_call_model is False


def test_build_rag_performance_protection_report_recommends_timeout_and_degradation() -> None:
    timing = assess_operation_timing(
        RagOperationStage.VECTOR_STORE,
        elapsed_ms=1200,
        timeout_seconds=1,
    )
    decision = choose_degradation_decision(
        RagOperationStage.VECTOR_STORE,
        has_cached_retrieval=True,
        has_safe_chunks=True,
    )

    report = build_rag_performance_protection_report(
        timings=[timing],
        degradation_decision=decision,
    )
    lines = format_rag_performance_protection_report(report)

    assert report.timed_out_stages == [RagOperationStage.VECTOR_STORE]
    assert report.degradation_mode is RagDegradationMode.USE_CACHED_RETRIEVAL
    assert report.high_priority_count >= 2
    assert any("timeout" in line for line in lines)
    assert any("cache" in line for line in lines)


def test_build_rag_performance_protection_report_recommends_cache_review_for_low_hit_rate() -> None:
    report = build_rag_performance_protection_report(
        cache_stats=RagCacheStats(
            hit_count=1,
            miss_count=9,
            set_count=10,
            evicted_count=0,
            current_entries=3,
        )
    )

    assert report.cache_hit_rate == 0.1
    assert any(
        recommendation.area == "cache"
        and "hit rate is low" in recommendation.reason
        for recommendation in report.recommendations
    )


def test_build_rag_performance_protection_report_recommends_stage_review_for_near_timeout() -> None:
    report = build_rag_performance_protection_report(
        timings=[
            assess_operation_timing(
                RagOperationStage.RERANK,
                elapsed_ms=850,
                timeout_seconds=1,
            )
        ]
    )

    assert report.near_timeout_stages == [RagOperationStage.RERANK]
    assert any(
        recommendation.area == "rerank"
        and recommendation.priority == "medium"
        for recommendation in report.recommendations
    )


def test_build_rag_performance_protection_report_recommends_safe_fallback_message() -> None:
    decision = choose_degradation_decision(
        RagOperationStage.GENERATION,
        has_cached_retrieval=False,
        has_safe_chunks=True,
    )

    report = build_rag_performance_protection_report(
        degradation_decision=decision,
    )

    assert report.degradation_mode is RagDegradationMode.RETURN_SAFE_FALLBACK
    assert any(
        recommendation.area == "degradation"
        and "safe retrieved evidence" in recommendation.reason
        for recommendation in report.recommendations
    )
