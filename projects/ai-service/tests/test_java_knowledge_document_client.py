import httpx
import pytest

from app.core.config import Settings
from app.services.java_knowledge_document_client import KnowledgeDocumentClient


def _make_settings() -> Settings:
    return Settings(
        _env_file=None,
        java_business_service_base_url="http://java.test",
        java_business_service_internal_token="local-dev-internal-token",
    )


def _make_client(handler) -> KnowledgeDocumentClient:
    return KnowledgeDocumentClient(
        base_url="http://java.test",
        timeout_seconds=2.0,
        settings=_make_settings(),
        transport=httpx.MockTransport(handler),
    )


def test_upsert_document_unwraps_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/knowledge-documents"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "success": True,
                "code": "OK",
                "data": {
                    "document_id": "doc-001",
                    "title": "Test",
                    "chunk_count": 3,
                },
            },
        )

    client = _make_client(handler)
    result = client.upsert_document(
        {
            "document_id": "doc-001",
            "title": "Test",
            "chunk_count": 3,
            "updated_by": "ai-service",
        }
    )
    assert result["document_id"] == "doc-001"
    assert result["chunk_count"] == 3


def test_delete_document_returns_bool() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/knowledge-documents/doc-001"
        assert request.method == "DELETE"
        return httpx.Response(200, json={"success": True, "code": "OK", "data": True})

    client = _make_client(handler)
    assert client.delete_document("doc-001") is True


def test_java_error_becomes_app_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "success": False,
                "code": "DOCUMENT_TITLE_REQUIRED",
                "message": "知识文档标题不能为空。",
            },
        )

    from app.core.exceptions import AppException

    client = _make_client(handler)
    with pytest.raises(AppException):
        client.upsert_document({"document_id": "doc-001"})
