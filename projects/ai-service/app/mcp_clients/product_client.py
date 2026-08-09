"""Product-grade MCP client used by the ticket agent to call the product MCP server."""

import asyncio
import json
import logging
from typing import Any, Callable, Protocol

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.exceptions import MCPError

from app.core.config import Settings, get_settings
from app.core.exceptions import AppException


logger = logging.getLogger(__name__)

# Reserved JSON-RPC server error code that the product MCP server's BearerAuth
# middleware returns in a JSON-RPC error body on authentication failure
# (401/403). The MCP client SDK surfaces it as MCPError with this code, which
# lets us distinguish "token wrong / server unauthenticated" from a transient
# server outage and fail fast without retrying.
MCP_AUTH_FAILED_ERROR_CODE = -32001
MCP_AUTH_FAILED_MESSAGE = "AI 工具服务认证失败，请检查 MCP_PRODUCT_AUTH_TOKEN 配置。"


MAX_LAST_ERROR_LOG_CHARS = 300


def _is_mcp_auth_failure(exc: Exception) -> bool:
    """True when the exception chain reports the server's auth failure code.

    The MCP SDK can wrap errors in an anyio ExceptionGroup (e.g. the GET SSE
    back-channel task), so recurse into group members defensively.
    """
    if isinstance(exc, MCPError) and exc.code == MCP_AUTH_FAILED_ERROR_CODE:
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_mcp_auth_failure(sub) for sub in exc.exceptions)
    return False


class McpToolCaller(Protocol):
    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return its parsed result dict."""
        ...


class ProductMcpClient:
    """Synchronous facade over the async streamable HTTP MCP client."""

    def __init__(
        self,
        base_url: str,
        auth_token: str | None,
        *,
        timeout_seconds: float = 30,
        retry_count: int = 2,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self._tools_cache: list[str] | None = None

    def _headers(self) -> dict[str, str]:
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        return {}

    def _build_http_client(self) -> httpx2.AsyncClient:
        return create_mcp_http_client(
            headers=self._headers(),
            timeout=httpx2.Timeout(self.timeout_seconds),
        )

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from app.agents.tracing_spans import start_tool_span

        with start_tool_span(tool_name=tool_name):
            return self._with_error_mapping(
                operation=f"call_tool:{tool_name}",
                fn=lambda: asyncio.run(self._call_tool_async(tool_name, arguments)),
            )

    def _with_error_mapping(
        self,
        operation: str,
        fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run an MCP call with the shared retry and error-mapping policy.

        - ``AppException`` propagates immediately (no retry).
        - Timeout errors map to ``MCP_SERVER_TIMEOUT`` (504) immediately.
        - Any other exception is retried up to ``retry_count`` times, then
          mapped to ``MCP_SERVER_UNREACHABLE`` (502).
        """
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                return fn()
            except AppException:
                raise
            except Exception as exc:
                last_error = exc
                if isinstance(exc, (TimeoutError, httpx2.TimeoutException)):
                    raise AppException(
                        code="MCP_SERVER_TIMEOUT",
                        message="AI 工具服务响应超时，请稍后再试。",
                        status_code=504,
                    ) from exc
                if _is_mcp_auth_failure(exc):
                    # Authentication is a configuration problem, not a transient
                    # outage: fail fast instead of burning retries against a
                    # server that will keep rejecting us.
                    raise AppException(
                        code="MCP_AUTH_FAILED",
                        message=MCP_AUTH_FAILED_MESSAGE,
                        status_code=502,
                    ) from exc
                logger.warning(
                    "product_mcp_client_retry operation=%s attempt=%s error_type=%s",
                    operation,
                    attempt + 1,
                    type(exc).__name__,
                )
        logger.error(
            "product_mcp_client_failed operation=%s error_type=%s error_message=%s",
            operation,
            type(last_error).__name__,
            str(last_error)[:MAX_LAST_ERROR_LOG_CHARS],
        )
        raise AppException(
            code="MCP_SERVER_UNREACHABLE",
            message="AI 工具服务暂时不可用，请稍后再试。",
            status_code=502,
        ) from last_error

    async def _call_tool_async(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        http_client = self._build_http_client()
        async with http_client:
            async with streamable_http_client(
                self.base_url,
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return self._parse_result(result)

    def _collect_text(self, result: Any) -> str:
        text_parts: list[str] = []
        for block in getattr(result, "content", []):
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return "\n".join(text_parts).strip()

    def _parse_result(self, result: Any) -> dict[str, Any]:
        if getattr(result, "is_error", getattr(result, "isError", False)):
            error_text = self._collect_text(result)
            raise AppException(
                code="MCP_TOOL_ERROR",
                message=(error_text or "AI 工具服务返回了错误。")[:200],
                status_code=502,
            )
        raw_text = self._collect_text(result)
        if not raw_text:
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的结果。",
                status_code=502,
            )
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的结果。",
                status_code=502,
            ) from exc
        if not isinstance(parsed, dict):
            raise AppException(
                code="MCP_RESULT_INVALID",
                message="AI 工具服务返回了无法解析的结果。",
                status_code=502,
            )
        return parsed

    def list_tools(self) -> list[str]:
        if self._tools_cache is not None:
            return self._tools_cache
        tools = self._with_error_mapping(
            operation="list_tools",
            fn=lambda: asyncio.run(self._list_tools_async()),
        )
        self._tools_cache = tools
        return tools

    async def _list_tools_async(self) -> list[str]:
        http_client = self._build_http_client()
        async with http_client:
            async with streamable_http_client(
                self.base_url,
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    return [tool.name for tool in listed.tools]


def create_product_mcp_client(
    settings: Settings | None = None,
) -> ProductMcpClient:
    resolved_settings = settings or get_settings()
    return ProductMcpClient(
        base_url=resolved_settings.resolved_mcp_product_base_url,
        auth_token=resolved_settings.resolved_mcp_product_auth_token,
        timeout_seconds=resolved_settings.mcp_product_timeout_seconds,
        retry_count=resolved_settings.mcp_product_retry_count,
    )
