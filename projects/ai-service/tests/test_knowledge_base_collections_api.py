from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.trace import TRACE_ID_HEADER
from app.rag.knowledge_routing import default_rag_knowledge_bases
from app.rag.vector_store import QdrantVectorStore


class FakeMultiStore:
    """模拟 QdrantVectorStore 的 list_collections / count_points。"""

    def __init__(self, existing: set[str]) -> None:
        self._existing = existing
        self.calls: list[str] = []

    def list_collections(self) -> list[str]:
        self.calls.append("list")
        return sorted(self._existing)

    def count_points(self) -> int:
        self.calls.append("count")
        return 3


def _patch_stores(monkeypatch, existing: set[str]) -> FakeMultiStore:
    fake = FakeMultiStore(existing)

    def fake_from_settings(settings, *, collection_name=None):
        return fake

    monkeypatch.setattr(
        QdrantVectorStore, "from_settings", staticmethod(fake_from_settings)
    )
    return fake


def test_collections_endpoint_lists_managed_collections(
    app: FastAPI,
    client: TestClient,
    monkeypatch,
) -> None:
    definitions = default_rag_knowledge_bases()
    managed_names = {d.collection_name for d in definitions}
    _patch_stores(monkeypatch, managed_names)

    response = client.get(
        "/api/knowledge-base/collections",
        headers={TRACE_ID_HEADER: "trace-kb-collections"},
    )
    assert response.status_code == 200
    data = response.json()
    names = [c["collection_name"] for c in data["collections"]]
    assert "kb_customer_policy" in names
    assert "kb_account_security" in names
    assert all(c["exists"] for c in data["collections"])
    assert all(c["point_count"] == 3 for c in data["collections"])
    assert all(not c["is_legacy"] for c in data["collections"])
    assert data["trace_id"] == "trace-kb-collections"


def test_collections_endpoint_marks_missing_and_legacy(
    app: FastAPI,
    client: TestClient,
    monkeypatch,
) -> None:
    definitions = default_rag_knowledge_bases()
    managed_names = {d.collection_name for d in definitions}
    # 只有 1 个 managed collection 存在；旧库 learning_rag_chunks_v4_1024 不存在
    _patch_stores(monkeypatch, {sorted(managed_names)[0]})

    response = client.get(
        "/api/knowledge-base/collections",
        headers={TRACE_ID_HEADER: "trace-kb-collections-2"},
    )
    assert response.status_code == 200
    data = response.json()
    by_name = {c["collection_name"]: c for c in data["collections"]}
    missing = [c for c in data["collections"] if not c["exists"]]
    assert len(missing) == len(managed_names) - 1
    assert all(c["point_count"] == 0 for c in missing)
    # legacy 展示（settings 默认 learning_rag_chunks）
    assert any(c["is_legacy"] for c in data["legacy_collections"])
