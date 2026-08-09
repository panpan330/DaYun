import pytest

from app.core.exceptions import AppException
from app.rag.filters import RagAccessScope
from app.rag.retriever import format_retrieved_chunks_for_debug, retrieve_top_k
from app.rag.vector_store import QdrantVectorStoreError
from tests.rag_fakes import (
    FakeEmbeddingModel,
    FakeVectorStoreReader,
    make_retrieved_chunk,
)


def test_retrieve_top_k_embeds_query_and_calls_vector_store() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    chunks = retrieve_top_k(
        "  订单多久发货？  ",
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "order_shipping_policy_chunk_0001"
    assert embedding_model.last_texts == ["订单多久发货？"]
    assert len(vector_store.last_call["query_vector"]) == 4
    assert vector_store.last_call["top_k"] == 3
    assert vector_store.last_call["payload_filter"] is None
    assert vector_store.last_call["score_threshold"] is None
    assert vector_store.last_call["with_payload"] is True
    assert vector_store.last_call["with_vector"] is False


def test_retrieve_top_k_rejects_blank_query() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    with pytest.raises(ValueError, match="query"):
        retrieve_top_k(
            "   ",
            embedding_model=embedding_model,
            vector_store=vector_store,
        )


def test_retrieve_top_k_rejects_invalid_top_k() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    with pytest.raises(ValueError, match="top_k"):
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=0,
        )


def test_retrieve_top_k_passes_payload_filter_to_vector_store() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    retrieve_top_k(
        "订单多久发货？",
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
        permission_group="customer_service",
        business_domain="order",
        doc_type="policy",
        source="order-shipping-policy.md",
    )

    assert vector_store.last_call["payload_filter"] == {
        "must": [
            {"key": "permission_group", "match": {"value": "customer_service"}},
            {"key": "business_domain", "match": {"value": "order"}},
            {"key": "doc_type", "match": {"value": "policy"}},
            {"key": "source", "match": {"value": "order-shipping-policy.md"}},
        ]
    }


def test_retrieve_top_k_accepts_access_scope_filter() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    retrieve_top_k(
        "閫€娆惧涔呭埌璐︼紵",
        embedding_model=embedding_model,
        vector_store=vector_store,
        access_scope=RagAccessScope(
            user_id="U1001",
            tenant_id="default",
            permission_groups=["customer_service"],
            business_domains=["refund"],
            excluded_statuses=["archived"],
        ),
    )

    assert vector_store.last_call["payload_filter"] == {
        "must": [
            {"key": "tenant_id", "match": {"value": "default"}},
            {"key": "permission_group", "match": {"any": ["customer_service"]}},
            {"key": "business_domain", "match": {"any": ["refund"]}},
        ],
        "must_not": [
            {"key": "status", "match": {"any": ["archived"]}},
        ],
    }


def test_retrieve_top_k_passes_score_threshold_to_vector_store() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    retrieve_top_k(
        "订单多久发货？",
        embedding_model=embedding_model,
        vector_store=vector_store,
        score_threshold=0.78,
    )

    assert vector_store.last_call["score_threshold"] == 0.78


def test_retrieve_top_k_rejects_blank_filter_value() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    with pytest.raises(ValueError, match="business_domain"):
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=embedding_model,
            vector_store=vector_store,
            business_domain="   ",
        )


def test_retrieve_top_k_rejects_invalid_score_threshold() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreReader()

    with pytest.raises(ValueError, match="score_threshold"):
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=embedding_model,
            vector_store=vector_store,
            score_threshold=True,
        )


def test_retrieve_top_k_rejects_bad_embedding_count() -> None:
    with pytest.raises(AppException) as exc_info:
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=FakeEmbeddingModel(dimension=4, vectors=[]),
            vector_store=FakeVectorStoreReader(),
        )

    assert exc_info.value.code == "RAG_EMBEDDING_BAD_RESPONSE"


def test_retrieve_top_k_rejects_bad_embedding_dimension() -> None:
    with pytest.raises(AppException) as exc_info:
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=FakeEmbeddingModel(dimension=4, vectors=[[0.1, 0.2]]),
            vector_store=FakeVectorStoreReader(),
        )

    assert exc_info.value.code == "RAG_EMBEDDING_BAD_RESPONSE"


def test_retrieve_top_k_maps_embedding_provider_error() -> None:
    with pytest.raises(AppException) as exc_info:
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=FakeEmbeddingModel(
                dimension=4,
                error=RuntimeError("embedding provider failed"),
            ),
            vector_store=FakeVectorStoreReader(),
        )

    assert exc_info.value.code == "RAG_EMBEDDING_FAILED"


def test_retrieve_top_k_maps_vector_store_error() -> None:
    with pytest.raises(AppException) as exc_info:
        retrieve_top_k(
            "订单多久发货？",
            embedding_model=FakeEmbeddingModel(dimension=4),
            vector_store=FakeVectorStoreReader(
                error=QdrantVectorStoreError("qdrant unavailable"),
            ),
        )

    assert exc_info.value.code == "RAG_VECTOR_STORE_FAILED"


def test_format_retrieved_chunks_for_debug() -> None:
    lines = format_retrieved_chunks_for_debug(
        [
            make_retrieved_chunk(
                metadata={
                    "source": "order-shipping-policy.md",
                    "section": "正常发货时效",
                },
                score=0.91234,
            )
        ]
    )

    assert lines == [
        "1. score=0.9123 source=order-shipping-policy.md section=正常发货时效 chunk_id=order_shipping_policy_chunk_0001"
    ]
