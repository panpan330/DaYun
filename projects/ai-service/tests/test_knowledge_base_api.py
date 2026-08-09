from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.trace import TRACE_ID_HEADER
from app.routers.knowledge_base import get_knowledge_base_dir, get_vector_store
from tests.rag_fakes import FakeVectorStoreWriter


def test_knowledge_base_status_returns_local_documents(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/knowledge-base/status",
        headers={TRACE_ID_HEADER: "trace-kb-status"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["document_count"] >= 4
    assert data["collection_name"] == "learning_rag_chunks"
    assert data["fake_embedding_dimension"] == 8
    assert data["real_embedding_configured"] is False
    assert data["trace_id"] == "trace-kb-status"
    assert {
        document["source"]
        for document in data["documents"]
    } >= {
        "account-security-faq.md",
        "refund-return-policy.md",
    }


def test_knowledge_base_ingest_uses_fake_embedding_and_vector_store(
    app: FastAPI,
    client: TestClient,
) -> None:
    vector_store = FakeVectorStoreWriter()
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    response = client.post(
        "/api/knowledge-base/ingest",
        headers={TRACE_ID_HEADER: "trace-kb-ingest"},
        json={
            "embedding_mode": "fake",
            "refresh": True,
            "wait": False,
            "chunk_size": 220,
            "chunk_overlap": 40,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["embedding_mode"] == "fake"
    assert data["document_count"] >= 4
    assert data["chunk_count"] == len(vector_store.embedded_chunks)
    assert data["vector_count"] == len(vector_store.embedded_chunks)
    assert data["vector_dimension"] == 8
    assert data["collection_name"] == "fake_chunks"
    assert data["replaced_source_count"] == len(vector_store.delete_calls)
    assert data["trace_id"] == "trace-kb-ingest"
    assert vector_store.last_ensure_call == {
        "vector_size": 8,
        "distance": "Cosine",
    }
    assert vector_store.last_upsert_call["wait"] is False


def test_knowledge_base_ingest_can_insert_without_refresh(
    app: FastAPI,
    client: TestClient,
) -> None:
    vector_store = FakeVectorStoreWriter()
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    response = client.post(
        "/api/knowledge-base/ingest",
        json={
            "embedding_mode": "fake",
            "refresh": False,
            "chunk_size": 220,
            "chunk_overlap": 40,
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert data["replaced_source_count"] == 0
    assert vector_store.delete_calls == []


def test_knowledge_base_ingest_rejects_real_embedding_without_key(
    app: FastAPI,
    client: TestClient,
) -> None:
    settings = Settings(_env_file=None)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_vector_store] = lambda: FakeVectorStoreWriter()

    response = client.post(
        "/api/knowledge-base/ingest",
        headers={TRACE_ID_HEADER: "trace-kb-real-no-key"},
        json={"embedding_mode": "real"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "EMBEDDING_API_KEY_MISSING",
        "message": "Embedding API key 未配置，无法执行真实 embedding 入库。",
        "trace_id": "trace-kb-real-no-key",
    }


def test_knowledge_base_status_rejects_missing_directory(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_knowledge_base_dir] = lambda: tmp_path / "missing"

    response = client.get(
        "/api/knowledge-base/status",
        headers={TRACE_ID_HEADER: "trace-kb-missing-dir"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "KNOWLEDGE_BASE_DIR_NOT_FOUND"
