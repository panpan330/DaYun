from collections.abc import Mapping
import logging
from typing import Any

import httpx

from app.core.config import Settings
from app.core.business_context import build_java_internal_headers
from app.core.exceptions import AppException
from app.core.trace import build_trace_headers
from app.services.java_error_mapping import build_java_error_app_exception
from app.services.java_order_client import _unwrap_java_api_response_data


logger = logging.getLogger(__name__)


class KnowledgeDocumentClient:
    """调 Java 内部接口持久化 knowledge_documents 元数据（仿 JavaOrderClient）。"""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.settings = settings
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeDocumentClient":
        return cls(
            base_url=settings.resolved_java_business_service_base_url,
            timeout_seconds=settings.resolved_java_business_service_timeout_seconds,
            settings=settings,
        )

    def list_documents(self) -> list[dict]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.get(
                    "/internal/knowledge-documents",
                    headers=self._build_headers(),
                )
            if response.status_code != 200:
                raise build_java_error_app_exception(
                    response,
                    operation="knowledge_document_list",
                    fallback_code="TOOL_UPSTREAM_ERROR",
                    fallback_message="知识文档列表加载失败，请稍后重试。",
                    fallback_status_code=502,
                )
            payload = response.json()
            if payload.get("success") is True:
                data = payload.get("data")
                return list(data) if isinstance(data, list) else []
            raise build_java_error_app_exception(
                response,
                operation="knowledge_document_list",
                fallback_code="TOOL_UPSTREAM_ERROR",
                fallback_message="知识文档列表加载失败。",
                fallback_status_code=502,
            )
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="知识文档列表加载失败。",
                status_code=502,
            ) from exc

    def upsert_document(self, payload: dict) -> Mapping[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.post(
                    "/internal/knowledge-documents",
                    json=payload,
                    headers=self._build_headers(),
                )
            if response.status_code != 200:
                raise build_java_error_app_exception(
                    response,
                    operation="knowledge_document_upsert",
                    fallback_code="TOOL_UPSTREAM_ERROR",
                    fallback_message="知识文档元数据保存失败，请稍后重试。",
                    fallback_status_code=502,
                )
            return _unwrap_java_api_response_data(response.json())
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="知识文档元数据保存失败。",
                status_code=502,
            ) from exc

    def delete_document(self, document_id: str) -> bool:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.delete(
                    f"/internal/knowledge-documents/{document_id}",
                    headers=self._build_headers(),
                )
            if response.status_code != 200:
                raise build_java_error_app_exception(
                    response,
                    operation="knowledge_document_delete",
                    fallback_code="TOOL_UPSTREAM_ERROR",
                    fallback_message="知识文档元数据删除失败，请稍后重试。",
                    fallback_status_code=502,
                )
            payload = response.json()
            if payload.get("success") is True:
                data = payload.get("data")
                if isinstance(data, bool):
                    return data
                return True
            raise build_java_error_app_exception(
                response,
                operation="knowledge_document_delete",
                fallback_code="TOOL_UPSTREAM_ERROR",
                fallback_message="知识文档元数据删除失败。",
                fallback_status_code=502,
            )
        except AppException:
            raise
        except Exception as exc:
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="知识文档元数据删除失败。",
                status_code=502,
            ) from exc

    def _build_headers(self) -> dict[str, str]:
        headers = build_trace_headers()
        if self.settings is not None:
            headers.update(build_java_internal_headers(self.settings))
        return headers
