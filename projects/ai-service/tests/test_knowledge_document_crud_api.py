from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.trace import TRACE_ID_HEADER
from app.routers.knowledge_base import get_knowledge_base_dir, get_vector_store
from app.services.java_knowledge_document_client import KnowledgeDocumentClient
from tests.rag_fakes import FakeVectorStoreWriter


class FakeKnowledgeDocumentClient:
    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []
        self.delete_calls: list[str] = []
        self.docs: list[dict] = []

    def list_documents(self) -> list[dict]:
        return list(self.docs)

    def upsert_document(self, payload: dict) -> dict:
        self.upsert_calls.append(payload)
        self.docs = [
            d for d in self.docs if d.get("document_id") != payload["document_id"]
        ]
        self.docs.append(payload)
        return {**payload, "updated_at": None}

    def delete_document(self, document_id: str) -> bool:
        self.delete_calls.append(document_id)
        self.docs = [d for d in self.docs if d.get("document_id") != document_id]
        return True


def _override_dependencies(
    app: FastAPI,
    tmp_path: Path,
    client: TestClient,
    monkeypatch=None,
) -> tuple[FakeKnowledgeDocumentClient, FakeVectorStoreWriter, FakeKnowledgeDocumentClient]:
    app.dependency_overrides[get_knowledge_base_dir] = lambda: tmp_path
    vector_store = FakeVectorStoreWriter()
    app.dependency_overrides[get_vector_store] = lambda: vector_store

    fake_java = FakeKnowledgeDocumentClient()
    from app.routers import knowledge_base as kb_router

    kb_router.build_java_document_client = lambda settings: fake_java
    if monkeypatch is not None:
        monkeypatch.setattr(kb_router, "build_collection_vector_store", lambda settings, collection_name=None: vector_store)
    return fake_java, vector_store, fake_java


def test_create_document_writes_file_and_syncs(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_java, vector_store, _ = _override_dependencies(app, tmp_path, client, monkeypatch)

    response = client.post(
        "/api/knowledge-base/documents",
        headers={TRACE_ID_HEADER: "trace-kb-create"},
        json={
            "document_id": "doc-001",
            "title": "Test Policy",
            "content": "# Test Policy\n\n退款政策七天无理由。",
            "business_domain": "refund",
            "permission_group": "public",
            "doc_type": "policy",
            "collection_name": "kb_customer_policy",
            "embedding_mode": "fake",
            "chunk_size": 220,
            "chunk_overlap": 40,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-001"
    assert (tmp_path / "doc-001.md").exists()
    assert fake_java.upsert_calls
    assert fake_java.upsert_calls[0]["document_id"] == "doc-001"
    assert vector_store.embedded_chunks  # Qdrant 同步发生


def test_update_document_resyncs_and_deletes_old_chunks(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_java, vector_store, _ = _override_dependencies(app, tmp_path, client, monkeypatch)

    create = client.post(
        "/api/knowledge-base/documents",
        headers={TRACE_ID_HEADER: "trace-kb-create"},
        json={
            "document_id": "doc-002",
            "title": "Old Title",
            "content": "# Old\n\n旧内容。",
            "business_domain": "refund",
            "permission_group": "public",
            "doc_type": "policy",
            "collection_name": "kb_customer_policy",
            "embedding_mode": "fake",
        },
    )
    assert create.status_code == 200
    old_delete_calls = len(vector_store.delete_calls)

    update = client.put(
        "/api/knowledge-base/documents/doc-002",
        headers={TRACE_ID_HEADER: "trace-kb-update"},
        json={
            "title": "New Title",
            "content": "# New\n\n新内容。",
            "embedding_mode": "fake",
        },
    )
    assert update.status_code == 200
    assert len(vector_store.delete_calls) > old_delete_calls  # 旧 chunk 被删
    assert vector_store.embedded_chunks  # 新 chunk upsert
    assert fake_java.upsert_calls[-1]["chunk_count"] >= 0


def test_delete_document_removes_file_and_metadata(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_java, vector_store, _ = _override_dependencies(app, tmp_path, client, monkeypatch)

    create = client.post(
        "/api/knowledge-base/documents",
        headers={TRACE_ID_HEADER: "trace-kb-create"},
        json={
            "document_id": "doc-003",
            "title": "To Delete",
            "content": "# To Delete\n\n内容。",
            "business_domain": "refund",
            "permission_group": "public",
            "doc_type": "policy",
            "collection_name": "kb_customer_policy",
            "embedding_mode": "fake",
        },
    )
    assert create.status_code == 200

    delete = client.delete(
        "/api/knowledge-base/documents/doc-003",
        headers={TRACE_ID_HEADER: "trace-kb-delete"},
    )
    assert delete.status_code == 200
    assert not (tmp_path / "doc-003.md").exists()
    assert fake_java.delete_calls == ["doc-003"]
    assert vector_store.delete_calls  # Qdrant chunk 删除


def test_list_documents_merges_local_and_java_metadata(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_java, _, _ = _override_dependencies(app, tmp_path, client, monkeypatch)
    (tmp_path / "existing.md").write_text(
        "# Existing Doc\n\n内容。\n文档类型: policy\n业务领域: refund\n权限组: public\n",
        encoding="utf-8",
    )
    # java 元数据预置一条（本地无对应文件 → exists_local=False）
    fake_java.docs.append(
        {
            "document_id": "java-only-doc",
            "title": "Java Only Doc",
            "business_domain": "account",
            "permission_group": "customer_service",
            "doc_type": "faq",
            "status": "enabled",
            "source_file_name": "java-only-doc.md",
            "chunk_count": 7,
            "updated_at": "2026-08-07T00:00:00",
        }
    )

    response = client.get(
        "/api/knowledge-base/documents",
        headers={TRACE_ID_HEADER: "trace-kb-list"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_count"] >= 1
    java_only = next(
        (d for d in data["documents"] if d["document_id"] == "java-only-doc"), None
    )
    assert java_only is not None
    assert java_only["chunk_count"] == 7
    assert java_only["exists_local"] is False
    assert any(d["source_file_name"] == "existing.md" for d in data["documents"])
    assert data["trace_id"] == "trace-kb-list"


def test_ingest_document_syncs_single_document(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_java, vector_store, _ = _override_dependencies(app, tmp_path, client, monkeypatch)
    (tmp_path / "doc-004.md").write_text(
        "# Doc Four\n\n内容。\n文档类型: policy\n业务领域: refund\n权限组: public\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/knowledge-base/documents/doc-004/ingest",
        headers={TRACE_ID_HEADER: "trace-kb-ingest-doc"},
        json={"embedding_mode": "fake"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "doc-004"
    assert data["chunk_count"] >= 1
    assert fake_java.upsert_calls
    assert fake_java.upsert_calls[-1]["document_id"] == "doc-004"


def test_document_id_rejects_unsafe_characters(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, _, _ = _override_dependencies(app, tmp_path, client, monkeypatch)
    # 分号不在白名单 → 422
    response = client.delete(
        "/api/knowledge-base/documents/evil;rm",
        headers={TRACE_ID_HEADER: "trace-kb-traversal"},
    )
    assert response.status_code == 422
    # 路径穿越字符（FastAPI 路由层 404 拦截，不进端点）
    traversal = client.delete(
        "/api/knowledge-base/documents/..%2F..%2Fevil",
        headers={TRACE_ID_HEADER: "trace-kb-traversal"},
    )
    assert traversal.status_code in (404, 422)
    assert traversal.status_code != 200


# ---- 文件上传（Task: knowledge base file upload）----


def _make_pdf_bytes() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "退款政策内容示例文本", fontsize=11, fontname="china-s")
    data = doc.tobytes()
    doc.close()
    return data


def _upload(
    client: TestClient,
    *,
    filename: str,
    content: bytes,
    data: dict | None = None,
):
    media_types = {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
    }
    suffix = Path(filename).suffix
    return client.post(
        "/api/knowledge-base/documents/upload",
        files={
            "file": (
                filename,
                content,
                media_types.get(suffix, "application/octet-stream"),
            )
        },
        data=data or {},
        headers={TRACE_ID_HEADER: "trace-upload"},
    )


def test_upload_markdown_document(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    fake_java, vector_store, _ = _override_dependencies(
        app, tmp_path, client, monkeypatch
    )
    response = _upload(
        client,
        filename="policy.md",
        content="# 退款政策\n七天无理由退货。".encode("utf-8"),
        data={"document_id": "doc-upload-md", "business_domain": "refund"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == "doc-upload-md"
    assert body["source_file_name"] == "doc-upload-md.md"
    assert (tmp_path / "doc-upload-md.md").exists()
    assert fake_java.upsert_calls, "java upsert 应被调用"
    assert vector_store.upsert_calls, "向量写入应发生"


def test_upload_txt_document(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    fake_java, _, _ = _override_dependencies(app, tmp_path, client, monkeypatch)
    response = _upload(
        client,
        filename="logistics.txt",
        content="物流政策内容".encode("utf-8"),
        data={"document_id": "doc-upload-txt"},
    )
    assert response.status_code == 200
    assert (tmp_path / "doc-upload-txt.txt").exists()
    assert fake_java.upsert_calls[0]["doc_type"] == "txt"


def test_upload_pdf_document(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    fake_java, _, _ = _override_dependencies(app, tmp_path, client, monkeypatch)
    response = _upload(
        client,
        filename="refund-policy.pdf",
        content=_make_pdf_bytes(),
        data={"document_id": "doc-upload-pdf"},
    )
    assert response.status_code == 200
    assert (tmp_path / "doc-upload-pdf.pdf").exists()
    assert fake_java.upsert_calls[0]["doc_type"] == "pdf"
    assert fake_java.upsert_calls[0]["title"], "PDF 提取出的标题不应为空"


def test_upload_rejects_unsupported_suffix(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    _override_dependencies(app, tmp_path, client, monkeypatch)
    response = _upload(client, filename="policy.docx", content=b"x")
    assert response.status_code == 400
    assert "docx" in response.json()["message"]


def test_upload_conflict_when_document_id_exists(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    _override_dependencies(app, tmp_path, client, monkeypatch)
    (tmp_path / "doc-dup.md").write_text("# a", encoding="utf-8")
    response = _upload(
        client,
        filename="other.md",
        content=b"# b",
        data={"document_id": "doc-dup"},
    )
    assert response.status_code == 409
    assert "已存在" in response.json()["message"]


def test_upload_uses_filename_stem_as_default_document_id(
    app: FastAPI, client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    _override_dependencies(app, tmp_path, client, monkeypatch)
    response = _upload(client, filename="auto-id.txt", content="内容".encode("utf-8"))
    assert response.status_code == 200
    assert response.json()["document_id"] == "auto-id"
