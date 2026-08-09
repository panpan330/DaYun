from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.performance import (
    InMemoryTtlCache,
    RagOperationStage,
    assess_operation_timing,
    build_batch_plan,
    build_retrieval_cache_key,
    choose_degradation_decision,
)


def main() -> None:
    query = "退款多久到账？"
    cache_key = build_retrieval_cache_key(
        query,
        top_k=5,
        score_threshold=0.7,
        permission_group="customer_service",
        embedding_model="text-embedding-demo",
        embedding_dimension=8,
        collection_name="learning_rag_chunks",
    )
    cache = InMemoryTtlCache(ttl_seconds=60, max_entries=2)
    cache.set(cache_key.value, ["refund_return_policy_chunk_0004"])

    batch_plan = build_batch_plan(
        [
            "chunk text 1",
            "chunk text 2",
            "chunk text 3",
            "chunk text 4",
            "chunk text 5",
        ],
        batch_size=2,
    )
    vector_timing = assess_operation_timing(
        RagOperationStage.VECTOR_STORE,
        elapsed_ms=850,
        timeout_seconds=1,
    )
    degradation = choose_degradation_decision(
        RagOperationStage.VECTOR_STORE,
        has_cached_retrieval=True,
        has_safe_chunks=False,
    )

    print("RAG performance preview")
    print(f"query: {query}")
    print(f"cache_key: {cache_key.value}")
    print(f"cache_hit: {cache.get(cache_key.value)}")
    print(f"cache_stats: {cache.stats().model_dump()}")
    print(f"batch_count: {batch_plan.batch_count} batches={batch_plan.batches}")
    print(
        "timing: "
        f"stage={vector_timing.stage.value} "
        f"elapsed_ms={vector_timing.elapsed_ms} "
        f"timeout_seconds={vector_timing.timeout_seconds} "
        f"status={vector_timing.status.value}"
    )
    print(
        "degradation: "
        f"mode={degradation.mode.value} "
        f"use_cache={degradation.should_use_cache} "
        f"call_model={degradation.should_call_model}"
    )


if __name__ == "__main__":
    main()
