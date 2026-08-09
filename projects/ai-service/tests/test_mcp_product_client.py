import httpx2
import pytest
from mcp.types import CallToolResult, TextContent

from app.core.config import Settings
from app.core.exceptions import AppException
from app.mcp_clients.product_client import (
    McpToolCaller,
    ProductMcpClient,
    create_product_mcp_client,
)
from tests.tool_fakes import FakeMcpToolCaller


class _FlakyCallTool:
    """Async _call_tool_async stand-in: raises ``error`` for the first
    ``fails`` calls, then returns ``result``."""

    def __init__(
        self,
        *,
        fails: int,
        error: Exception,
        result: dict | None = None,
    ) -> None:
        self.fails = fails
        self.error = error
        self.result = result or {"ok": True}
        self.calls = 0

    async def __call__(self, tool_name: str, arguments: dict) -> dict:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.error
        return self.result


class _TrackingListTools:
    """Async _list_tools_async stand-in that records how often it is invoked."""

    def __init__(
        self,
        result: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or ["query_order", "create_ticket"]
        self.error = error
        self.calls = 0

    async def __call__(self) -> list[str]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _text_result(*texts: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text) for text in texts],
        isError=False,
    )


def test_create_product_mcp_client_from_settings() -> None:
    client = create_product_mcp_client(
        Settings(
            _env_file=None,
            mcp_product_base_url="http://127.0.0.1:9100/mcp",
            mcp_product_auth_token="token",
            mcp_product_timeout_seconds=5,
            mcp_product_retry_count=1,
        )
    )
    assert isinstance(client, ProductMcpClient)
    assert client.base_url == "http://127.0.0.1:9100/mcp"
    assert client.auth_token == "token"
    assert client.timeout_seconds == 5
    assert client.retry_count == 1


def test_mcp_tool_caller_protocol_supports_fake() -> None:
    fake = FakeMcpToolCaller(responses={"query_order": {"ok": True}})
    caller: McpToolCaller = fake
    assert caller.call_tool("query_order", {"order_id": "A1001"}) == {"ok": True}
    assert fake.calls == [("query_order", {"order_id": "A1001"})]


def test_parse_result_returns_parsed_dict() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    result = _text_result('{"ok": true, "order_id": "A1001"}')
    assert client._parse_result(result) == {"ok": True, "order_id": "A1001"}


def test_parse_result_joins_multiple_text_blocks() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    result = _text_result('{"ok":', " true}")
    assert client._parse_result(result) == {"ok": True}


def test_parse_result_rejects_non_json_text() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    with pytest.raises(AppException) as exc_info:
        client._parse_result(_text_result("not json"))
    assert exc_info.value.code == "MCP_RESULT_INVALID"
    assert exc_info.value.status_code == 502


def test_parse_result_rejects_non_dict_json() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    with pytest.raises(AppException) as exc_info:
        client._parse_result(_text_result("[1, 2, 3]"))
    assert exc_info.value.code == "MCP_RESULT_INVALID"
    assert exc_info.value.status_code == 502


def test_parse_result_rejects_missing_text_blocks() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    with pytest.raises(AppException) as exc_info:
        client._parse_result(CallToolResult(content=[], isError=False))
    assert exc_info.value.code == "MCP_RESULT_INVALID"
    assert exc_info.value.status_code == 502


def test_parse_result_surfaces_tool_error_text() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text='{"error_code":"TOOL_TIMEOUT","detail":"upstream timed out"}',
            )
        ],
        isError=True,
    )
    with pytest.raises(AppException) as exc_info:
        client._parse_result(result)
    assert exc_info.value.code == "MCP_TOOL_ERROR"
    assert exc_info.value.status_code == 502
    assert "TOOL_TIMEOUT" in exc_info.value.message


def test_parse_result_tool_error_message_truncated() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    result = CallToolResult(
        content=[TextContent(type="text", text="x" * 500)],
        isError=True,
    )
    with pytest.raises(AppException) as exc_info:
        client._parse_result(result)
    assert exc_info.value.code == "MCP_TOOL_ERROR"
    assert len(exc_info.value.message) <= 200


def test_parse_result_tool_error_without_text_falls_back() -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    result = CallToolResult(content=[], isError=True)
    with pytest.raises(AppException) as exc_info:
        client._parse_result(result)
    assert exc_info.value.code == "MCP_TOOL_ERROR"
    assert exc_info.value.status_code == 502
    assert exc_info.value.message


def test_call_tool_retries_then_succeeds(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    flaky = _FlakyCallTool(fails=2, error=ConnectionError("boom"))
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    assert client.call_tool("query_order", {"order_id": "A1001"}) == {"ok": True}
    assert flaky.calls == 3


def test_call_tool_exhausts_retries_maps_unreachable(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    flaky = _FlakyCallTool(fails=99, error=ConnectionError("boom"))
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with pytest.raises(AppException) as exc_info:
        client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_SERVER_UNREACHABLE"
    assert exc_info.value.status_code == 502
    assert flaky.calls == 3


@pytest.mark.parametrize(
    "timeout_error",
    [
        TimeoutError("timeout"),
        httpx2.ReadTimeout("timeout"),
    ],
)
def test_call_tool_timeout_maps_to_504_without_retry(
    monkeypatch,
    timeout_error,
) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    flaky = _FlakyCallTool(fails=99, error=timeout_error)
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with pytest.raises(AppException) as exc_info:
        client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_SERVER_TIMEOUT"
    assert exc_info.value.status_code == 504
    assert flaky.calls == 1


def test_call_tool_propagates_app_exception_without_retry(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    flaky = _FlakyCallTool(
        fails=99,
        error=AppException(
            code="MCP_TOOL_NOT_FOUND",
            message="tool query_order not available",
            status_code=404,
        ),
    )
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with pytest.raises(AppException) as exc_info:
        client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_TOOL_NOT_FOUND"
    assert exc_info.value.status_code == 404
    assert flaky.calls == 1


def test_call_tool_auth_failure_maps_to_auth_failed_without_retry(
    monkeypatch,
) -> None:
    from mcp.shared.exceptions import MCPError

    from app.mcp_clients.product_client import MCP_AUTH_FAILED_ERROR_CODE

    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    flaky = _FlakyCallTool(
        fails=99,
        error=MCPError(
            code=MCP_AUTH_FAILED_ERROR_CODE,
            message="missing or invalid bearer token",
        ),
    )
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with pytest.raises(AppException) as exc_info:
        client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_AUTH_FAILED"
    assert exc_info.value.status_code == 502
    assert "MCP_PRODUCT_AUTH_TOKEN" in exc_info.value.message
    # Authentication failure is a configuration problem, not a transient
    # outage: no retries are burned.
    assert flaky.calls == 1


def test_call_tool_auth_failure_inside_exception_group_without_retry(
    monkeypatch,
) -> None:
    from mcp.shared.exceptions import MCPError

    from app.mcp_clients.product_client import MCP_AUTH_FAILED_ERROR_CODE

    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    inner = MCPError(code=MCP_AUTH_FAILED_ERROR_CODE, message="unauthorized")
    flaky = _FlakyCallTool(fails=99, error=BaseExceptionGroup("mcp", [inner]))
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with pytest.raises(AppException) as exc_info:
        client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_AUTH_FAILED"
    assert flaky.calls == 1


def test_call_tool_exhausted_logs_last_error_type_and_message(
    monkeypatch,
    caplog,
) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=1)
    flaky = _FlakyCallTool(fails=99, error=ConnectionError("boom"))
    monkeypatch.setattr(client, "_call_tool_async", flaky)
    with caplog.at_level("ERROR", logger="app.mcp_clients.product_client"):
        with pytest.raises(AppException) as exc_info:
            client.call_tool("query_order", {"order_id": "A1001"})
    assert exc_info.value.code == "MCP_SERVER_UNREACHABLE"
    log_lines = [
        record
        for record in caplog.records
        if record.getMessage().startswith("product_mcp_client_failed")
    ]
    assert len(log_lines) == 1
    assert "error_type=ConnectionError" in log_lines[0].getMessage()
    assert "error_message=boom" in log_lines[0].getMessage()


def test_list_tools_caches_second_call(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None)
    tracking = _TrackingListTools()
    monkeypatch.setattr(client, "_list_tools_async", tracking)
    assert client.list_tools() == ["query_order", "create_ticket"]
    assert client.list_tools() == ["query_order", "create_ticket"]
    assert tracking.calls == 1


def test_list_tools_failure_not_cached_and_maps_unreachable(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    tracking = _TrackingListTools()
    tracking.error = ConnectionError("boom")
    monkeypatch.setattr(client, "_list_tools_async", tracking)
    with pytest.raises(AppException) as exc_info:
        client.list_tools()
    assert exc_info.value.code == "MCP_SERVER_UNREACHABLE"
    assert exc_info.value.status_code == 502
    assert tracking.calls == 3
    assert client._tools_cache is None
    tracking.error = None
    assert client.list_tools() == ["query_order", "create_ticket"]
    assert tracking.calls == 4
    assert client._tools_cache == ["query_order", "create_ticket"]


def test_list_tools_timeout_maps_to_504(monkeypatch) -> None:
    client = ProductMcpClient("http://127.0.0.1:9100/mcp", None, retry_count=2)
    tracking = _TrackingListTools(error=TimeoutError("timeout"))
    monkeypatch.setattr(client, "_list_tools_async", tracking)
    with pytest.raises(AppException) as exc_info:
        client.list_tools()
    assert exc_info.value.code == "MCP_SERVER_TIMEOUT"
    assert exc_info.value.status_code == 504
    assert tracking.calls == 1
