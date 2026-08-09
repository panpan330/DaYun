from app.core.exceptions import AppException
from app.rag.errors import (
    rag_embedding_bad_response,
    rag_embedding_failed,
    rag_vector_store_failed,
)
from app.rag.vector_store import QdrantCollectionConfigError, QdrantVectorStoreError


def test_rag_embedding_failed_maps_unknown_error_to_app_exception() -> None:
    exc = rag_embedding_failed(RuntimeError("provider unavailable"))

    assert exc.code == "RAG_EMBEDDING_FAILED"
    assert exc.status_code == 502
    assert "embedding" in exc.message


def test_rag_embedding_failed_preserves_existing_app_exception() -> None:
    original = AppException(code="CUSTOM_RAG_ERROR", message="custom", status_code=503)

    assert rag_embedding_failed(original) is original


def test_rag_embedding_failed_maps_value_error_to_bad_response() -> None:
    exc = rag_embedding_failed(ValueError("vector size mismatch"))

    assert exc.code == "RAG_EMBEDDING_BAD_RESPONSE"
    assert exc.status_code == 502


def test_rag_embedding_bad_response() -> None:
    exc = rag_embedding_bad_response()

    assert exc.code == "RAG_EMBEDDING_BAD_RESPONSE"
    assert exc.status_code == 502


def test_rag_vector_store_failed_maps_qdrant_error() -> None:
    exc = rag_vector_store_failed(QdrantVectorStoreError("failed to connect"))

    assert exc.code == "RAG_VECTOR_STORE_FAILED"
    assert exc.status_code == 502
    assert "向量库" in exc.message


def test_rag_vector_store_failed_maps_collection_config_error() -> None:
    exc = rag_vector_store_failed(QdrantCollectionConfigError("dimension mismatch"))

    assert exc.code == "RAG_VECTOR_STORE_CONFIG_ERROR"
    assert exc.status_code == 500


def test_rag_vector_store_failed_preserves_existing_app_exception() -> None:
    original = AppException(code="CUSTOM_VECTOR_ERROR", message="custom", status_code=503)

    assert rag_vector_store_failed(original) is original
