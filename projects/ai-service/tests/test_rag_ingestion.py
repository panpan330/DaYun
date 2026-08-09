from pathlib import Path

import pytest

from app.core.exceptions import AppException
from app.rag.ingestion import (
    delete_document_from_vector_store,
    ingest_directory_to_vector_store,
    refresh_directory_in_vector_store,
)
from app.rag.vector_store import QdrantCollectionConfigError
from tests.rag_fakes import FakeEmbeddingModel, FakeVectorStoreWriter


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


def test_ingest_directory_loads_splits_embeds_and_writes_chunks() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreWriter()

    result = ingest_directory_to_vector_store(
        KNOWLEDGE_BASE_DIR,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=220,
        chunk_overlap=40,
    )

    assert result.document_count >= 1
    assert result.chunk_count == len(vector_store.embedded_chunks)
    assert result.vector_count == len(vector_store.embedded_chunks)
    assert result.vector_dimension == 4
    assert result.collection_name == "fake_chunks"
    assert vector_store.last_ensure_call["vector_size"] == 4
    assert vector_store.last_ensure_call["distance"] == "Cosine"
    assert vector_store.last_upsert_call["wait"] is True
    assert len(embedding_model.calls) == 1
    assert vector_store.embedded_chunks[0].metadata["chunk_id"]


def test_ingest_directory_maps_embedding_failure() -> None:
    with pytest.raises(AppException) as exc_info:
        ingest_directory_to_vector_store(
            KNOWLEDGE_BASE_DIR,
            embedding_model=FakeEmbeddingModel(
                dimension=4,
                error=RuntimeError("embedding provider failed"),
            ),
            vector_store=FakeVectorStoreWriter(),
            chunk_size=220,
            chunk_overlap=40,
        )

    assert exc_info.value.code == "RAG_EMBEDDING_FAILED"


def test_ingest_directory_maps_vector_store_config_error() -> None:
    with pytest.raises(AppException) as exc_info:
        ingest_directory_to_vector_store(
            KNOWLEDGE_BASE_DIR,
            embedding_model=FakeEmbeddingModel(dimension=4),
            vector_store=FakeVectorStoreWriter(
                ensure_error=QdrantCollectionConfigError("dimension mismatch"),
            ),
            chunk_size=220,
            chunk_overlap=40,
        )

    assert exc_info.value.code == "RAG_VECTOR_STORE_CONFIG_ERROR"


def test_refresh_directory_deletes_existing_sources_before_upsert() -> None:
    embedding_model = FakeEmbeddingModel(dimension=4)
    vector_store = FakeVectorStoreWriter()

    result = refresh_directory_in_vector_store(
        KNOWLEDGE_BASE_DIR,
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=220,
        chunk_overlap=40,
        wait=False,
    )

    embedded_sources = {
        chunk.metadata["source"]
        for chunk in vector_store.embedded_chunks
        if isinstance(chunk.metadata.get("source"), str)
    }
    deleted_sources = {
        call["payload_filter"]["must"][0]["match"]["value"]
        for call in vector_store.delete_calls
    }

    assert result.document_count >= 1
    assert result.replaced_source_count == len(deleted_sources)
    assert deleted_sources == embedded_sources
    assert vector_store.last_upsert_call["wait"] is False
    assert all(call["wait"] is False for call in vector_store.delete_calls)


def test_delete_document_from_vector_store_deletes_by_source() -> None:
    vector_store = FakeVectorStoreWriter()

    result = delete_document_from_vector_store(
        " shipping.md ",
        vector_store=vector_store,
        wait=False,
    )

    assert result.source == "shipping.md"
    assert result.collection_name == "fake_chunks"
    assert vector_store.last_delete_call == {
        "payload_filter": {
            "must": [{"key": "source", "match": {"value": "shipping.md"}}]
        },
        "wait": False,
    }


def test_refresh_directory_maps_delete_failure() -> None:
    with pytest.raises(AppException) as exc_info:
        refresh_directory_in_vector_store(
            KNOWLEDGE_BASE_DIR,
            embedding_model=FakeEmbeddingModel(dimension=4),
            vector_store=FakeVectorStoreWriter(
                delete_error=RuntimeError("delete failed")
            ),
            chunk_size=220,
            chunk_overlap=40,
        )

    assert exc_info.value.code == "RAG_VECTOR_STORE_FAILED"


def test_delete_document_from_vector_store_rejects_blank_source() -> None:
    with pytest.raises(ValueError, match="source"):
        delete_document_from_vector_store("   ", vector_store=FakeVectorStoreWriter())
