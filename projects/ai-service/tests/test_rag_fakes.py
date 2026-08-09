import pytest

from app.rag.embeddings import EmbeddedChunk
from tests.rag_fakes import (
    FakeEmbeddingModel,
    FakeVectorStoreReader,
    FakeVectorStoreWriter,
    make_retrieved_chunk,
)


def test_make_retrieved_chunk_allows_overrides() -> None:
    chunk = make_retrieved_chunk(
        chunk_id="refund_policy_chunk_0001",
        content="退款通常会在审核通过后处理。",
        score=0.82,
        metadata={"source": "refund.md", "section": "退款时效"},
    )

    assert chunk.chunk_id == "refund_policy_chunk_0001"
    assert chunk.content == "退款通常会在审核通过后处理。"
    assert chunk.metadata["source"] == "refund.md"
    assert chunk.score == 0.82


def test_fake_embedding_model_records_texts_and_returns_vectors() -> None:
    model = FakeEmbeddingModel(dimension=3)

    vectors = model.embed_texts(["订单发货", "退款规则"])

    assert model.last_texts == ["订单发货", "退款规则"]
    assert len(vectors) == 2
    assert all(len(vector) == 3 for vector in vectors)


def test_fake_embedding_model_can_return_configured_vectors() -> None:
    model = FakeEmbeddingModel(vectors=[[0.1, 0.2], [0.3, 0.4]])

    vectors = model.embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert model.last_texts == ["first", "second"]


def test_fake_embedding_model_can_raise_configured_error() -> None:
    model = FakeEmbeddingModel(error=RuntimeError("embedding failed"))

    with pytest.raises(RuntimeError, match="embedding failed"):
        model.embed_texts(["订单发货"])


def test_fake_vector_store_reader_records_query_arguments() -> None:
    store = FakeVectorStoreReader(chunks=[make_retrieved_chunk(chunk_id="chunk-1")])

    chunks = store.query_similar(
        [0.1, 0.2],
        top_k=2,
        payload_filter={"must": [{"key": "source", "match": {"value": "demo.md"}}]},
        score_threshold=0.75,
        with_payload=True,
        with_vector=False,
    )

    assert chunks[0].chunk_id == "chunk-1"
    assert store.last_call["query_vector"] == [0.1, 0.2]
    assert store.last_call["top_k"] == 2
    assert store.last_call["score_threshold"] == 0.75
    assert store.last_call["with_payload"] is True
    assert store.last_call["with_vector"] is False


def test_fake_vector_store_reader_can_raise_configured_error() -> None:
    store = FakeVectorStoreReader(error=RuntimeError("vector store failed"))

    with pytest.raises(RuntimeError, match="vector store failed"):
        store.query_similar([0.1, 0.2], top_k=1)


def test_fake_vector_store_reader_applies_top_k_and_score_threshold() -> None:
    store = FakeVectorStoreReader(
        chunks=[
            make_retrieved_chunk(chunk_id="chunk-low", score=0.42),
            make_retrieved_chunk(chunk_id="chunk-high", score=0.95),
            make_retrieved_chunk(chunk_id="chunk-mid", score=0.81),
        ]
    )

    chunks = store.query_similar(
        [0.1, 0.2],
        top_k=2,
        score_threshold=0.8,
    )

    assert [chunk.chunk_id for chunk in chunks] == ["chunk-high", "chunk-mid"]


def test_fake_vector_store_writer_records_collection_and_upsert_arguments() -> None:
    embedded = EmbeddedChunk(
        chunk_id="chunk-1",
        content="订单付款后 24 小时内发货。",
        metadata={"source": "shipping.md"},
        vector=[0.1, 0.2],
    )
    store = FakeVectorStoreWriter()

    store.ensure_collection(vector_size=2, distance="Cosine")
    count = store.upsert_embedded_chunks([embedded], wait=False)

    assert count == 1
    assert store.last_ensure_call == {"vector_size": 2, "distance": "Cosine"}
    assert store.last_upsert_call["embedded_chunks"] == [embedded]
    assert store.last_upsert_call["wait"] is False


def test_fake_vector_store_writer_records_delete_arguments() -> None:
    store = FakeVectorStoreWriter()

    store.delete_points_by_filter(
        {"must": [{"key": "source", "match": {"value": "shipping.md"}}]},
        wait=False,
    )

    assert store.last_delete_call == {
        "payload_filter": {
            "must": [{"key": "source", "match": {"value": "shipping.md"}}]
        },
        "wait": False,
    }


def test_fake_vector_store_writer_can_raise_configured_errors() -> None:
    ensure_error = RuntimeError("ensure failed")
    upsert_error = RuntimeError("upsert failed")
    delete_error = RuntimeError("delete failed")
    ensure_store = FakeVectorStoreWriter(ensure_error=ensure_error)
    upsert_store = FakeVectorStoreWriter(upsert_error=upsert_error)
    delete_store = FakeVectorStoreWriter(delete_error=delete_error)

    with pytest.raises(RuntimeError, match="ensure failed"):
        ensure_store.ensure_collection(vector_size=2)

    with pytest.raises(RuntimeError, match="upsert failed"):
        upsert_store.upsert_embedded_chunks([], wait=True)

    with pytest.raises(RuntimeError, match="delete failed"):
        delete_store.delete_points_by_filter(
            {"must": [{"key": "source", "match": {"value": "shipping.md"}}]}
        )
