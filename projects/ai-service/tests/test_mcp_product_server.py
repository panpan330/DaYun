from types import SimpleNamespace

import pytest
from mcp import Client

from app.core.business_context import build_java_internal_headers, get_business_context
from app.core.config import Settings
from app.core.exceptions import AppException
from app.mcp_servers import order_tool
from app.mcp_servers.product_server import (
    BearerAuthMiddleware,
    create_product_mcp_app,
    create_product_mcp_server,
)
from tests.tool_fakes import make_created_ticket


def _settings(token: str | None = "test-token") -> Settings:
    return Settings(
        _env_file=None,
        mcp_product_auth_token=token,
        tool_confirmation_backend="memory",
    )


def test_product_server_registers_only_business_tools() -> None:
    async def run() -> None:
        server = create_product_mcp_server(_settings())
        async with Client(server) as client:
            tools = await client.list_tools()
        names = [tool.name for tool in tools.tools]
        assert names == ["query_order", "create_ticket", "refund_order", "cancel_order"]

    import asyncio

    asyncio.run(run())


def test_product_create_ticket_accepts_16_to_32_hex_confirmation_id() -> None:
    server = create_product_mcp_server(_settings())

    async def run() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools.tools if t.name == "create_ticket")
        confirmation_schema = tool.input_schema["properties"]["confirmation_id"]
        assert confirmation_schema["pattern"] == r"^[a-f0-9]{16,32}$"
        assert "user_confirmed" in tool.input_schema["properties"]
        # The authenticated actor identity contract: the AI service injects
        # user_id/tenant_id into every business tool call so the MCP server
        # can set its business context before calling Java.
        assert "user_id" in tool.input_schema["properties"]
        assert "tenant_id" in tool.input_schema["properties"]

    import asyncio

    asyncio.run(run())


def test_product_query_order_contract_includes_identity_parameters() -> None:
    server = create_product_mcp_server(_settings())

    async def run() -> None:
        async with Client(server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools.tools if t.name == "query_order")
        assert "user_id" in tool.input_schema["properties"]
        assert "tenant_id" in tool.input_schema["properties"]

    import asyncio

    asyncio.run(run())


def test_product_query_order_sets_business_context_before_java_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_query_order_for_mcp(order_id: str, **_: object) -> dict[str, object]:
        captured["order_id"] = order_id
        captured["context"] = get_business_context()
        return {
            "ok": True,
            "result": {
                "order_id": order_id,
                "order_status": "waiting_shipment",
                "payment_status": "paid",
                "can_create_ticket": True,
                "source": "java_business_service",
            },
        }

    monkeypatch.setattr(order_tool, "query_order_for_mcp", fake_query_order_for_mcp)
    server = create_product_mcp_server(_settings())

    async def run() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "query_order",
                {
                    "order_id": "A1001",
                    "user_id": "U5000",
                    "tenant_id": "tenant-5",
                },
            )
        assert result.content  # tool executed successfully

    import asyncio

    asyncio.run(run())
    # The Java call ran with the caller's identity in the business context...
    assert captured["context"] == ("U5000", "tenant-5")
    # ...and the context was cleaned up afterwards (no cross-request leakage).
    assert get_business_context() == (None, None)


def test_product_create_ticket_sets_business_context_before_java_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    captured: dict[str, object] = {}

    class _ContextCapturingTicketCreator:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def create_ticket(
            self,
            arguments: object,
            *,
            idempotency_key: str,
        ) -> object:
            captured["headers"] = build_java_internal_headers(self.settings)
            return make_created_ticket()

    monkeypatch.setattr(
        product_server,
        "JavaTicketClient",
        SimpleNamespace(
            from_settings=staticmethod(
                lambda settings: _ContextCapturingTicketCreator(settings)
            )
        ),
    )

    class _ConfirmingStore:
        def require_confirmed(
            self,
            confirmation_id: str,
            *,
            actor_id: str,
        ) -> None:
            return None

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _ConfirmingStore(),
    )

    result = product_server._product_create_ticket(
        requester_id="user-1",
        title="测试工单",
        description="测试描述",
        category="refund",
        confirmation_id="a" * 32,
        user_confirmed=True,
        user_id="U5000",
        tenant_id="tenant-5",
    )
    assert result["ok"] is True
    # The Java ticket call saw the injected identity, not the default fallback.
    assert captured["headers"]["X-User-Id"] == "U5000"
    assert captured["headers"]["X-Tenant-Id"] == "tenant-5"
    # The business context was reset after the Java call.
    assert get_business_context() == (None, None)


def test_bearer_auth_middleware_rejects_missing_token() -> None:
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def ok_endpoint(request: object) -> PlainTextResponse:
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/mcp", ok_endpoint, methods=["GET", "POST"])])
    wrapped = BearerAuthMiddleware(inner, token="test-token")

    from starlette.testclient import TestClient

    client = TestClient(wrapped)
    response = client.post("/mcp", json={})
    assert response.status_code == 401
    # The 401 body is a JSON-RPC error carrying the reserved auth code so the
    # MCP client SDK surfaces it as MCP_AUTH_FAILED instead of retrying.
    from app.mcp_clients.product_client import MCP_AUTH_FAILED_ERROR_CODE

    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["error"]["code"] == MCP_AUTH_FAILED_ERROR_CODE


def test_bearer_auth_middleware_rejects_wrong_token() -> None:
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def ok_endpoint(request: object) -> PlainTextResponse:
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/mcp", ok_endpoint, methods=["GET", "POST"])])
    wrapped = BearerAuthMiddleware(inner, token="test-token")
    from starlette.testclient import TestClient

    client = TestClient(wrapped)
    response = client.post(
        "/mcp",
        json={},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_bearer_auth_middleware_accepts_correct_token() -> None:
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def ok_endpoint(request: object) -> PlainTextResponse:
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/mcp", ok_endpoint, methods=["GET", "POST"])])
    wrapped = BearerAuthMiddleware(inner, token="test-token")
    from starlette.testclient import TestClient

    client = TestClient(wrapped)
    response = client.post(
        "/mcp",
        json={},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200


def test_product_app_uses_bearer_middleware() -> None:
    app = create_product_mcp_app(_settings(token="test-token"))
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": {}},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code in (200, 202)


def test_product_app_allows_local_origin() -> None:
    app = create_product_mcp_app(_settings(token="test-token"))
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": {}},
        headers={
            "Authorization": "Bearer test-token",
            "Origin": "http://localhost:5173",
        },
    )
    assert response.status_code in (200, 202)


def test_product_app_rejects_non_local_origin() -> None:
    app = create_product_mcp_app(_settings(token="test-token"))
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1, "params": {}},
        headers={
            "Authorization": "Bearer test-token",
            "Origin": "https://evil.example.com",
        },
    )
    assert response.status_code == 403


def test_product_create_ticket_confirmation_unavailable_mapped_to_ok_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.mcp_servers import product_server

    class _FailingConfirmationStore:
        def require_confirmed(self, confirmation_id: str, *, actor_id: str) -> None:
            raise RedisConnectionError("redis connection refused")

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _FailingConfirmationStore(),
    )
    result = product_server._product_create_ticket(
        requester_id="user-1",
        title="测试工单",
        description="测试描述",
        category="refund",
        confirmation_id="a" * 32,
        user_confirmed=True,
    )
    assert result["ok"] is False
    assert result["error_code"] == "TOOL_CONFIRMATION_UNAVAILABLE"
    assert result["ticket"] is None


def test_product_refund_order_requires_user_confirmation() -> None:
    from app.mcp_servers import product_server

    result = product_server._product_refund_order(
        order_id="A1002",
        reason="七天无理由退货",
        confirmation_id="a" * 32,
        requester_id="user-1",
        user_confirmed=False,
    )

    assert result["ok"] is False
    assert result["error_code"] == "TOOL_CONFIRMATION_REQUIRED"
    assert result["refund"] is None


def test_product_refund_order_validates_confirmation_id_format() -> None:
    server = create_product_mcp_server(_settings())

    async def run() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "refund_order",
                {
                    "order_id": "A1002",
                    "reason": "七天无理由退货",
                    "requester_id": "user-1",
                    "confirmation_id": "not-a-hex-id",
                },
            )
        # MCP 2.0 surfaces pydantic argument-validation failures as an is_error
        # CallToolResult whose text describes the confirmation_id pattern.
        assert result.is_error is True
        error_text = result.content[0].text
        assert "confirmation_id" in error_text
        assert "string_pattern_mismatch" in error_text

    import asyncio

    asyncio.run(run())


def test_product_refund_order_sets_business_context_before_java_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    captured: dict[str, object] = {}

    class _ContextCapturingOrderClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def refund_order(
            self,
            order_id: str,
            reason: str,
            *,
            idempotency_key: str | None = None,
            trace_context: object = None,
        ) -> dict[str, object]:
            captured["order_id"] = order_id
            captured["reason"] = reason
            captured["idempotency_key"] = idempotency_key
            captured["headers"] = build_java_internal_headers(self.settings)
            return {
                "order_id": order_id,
                "payment_status": "refunded",
                "refund_amount": 159.00,
            }

    monkeypatch.setattr(
        product_server,
        "JavaOrderClient",
        SimpleNamespace(
            from_settings=staticmethod(
                lambda settings: _ContextCapturingOrderClient(settings)
            )
        ),
    )

    class _ConfirmingStore:
        def require_confirmed(
            self,
            confirmation_id: str,
            *,
            actor_id: str,
        ) -> None:
            captured["confirmation_actor_id"] = actor_id
            return None

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _ConfirmingStore(),
    )

    result = product_server._product_refund_order(
        order_id="A1002",
        reason="七天无理由退货",
        confirmation_id="b" * 32,
        requester_id="U5000",
        user_confirmed=True,
        user_id="U5000",
        tenant_id="tenant-5",
    )
    assert result["ok"] is True
    # The confirmation store verified the real requester id, not a demo fallback.
    assert captured["confirmation_actor_id"] == "U5000"
    # The Java refund call saw the injected identity, not the default fallback.
    assert captured["headers"]["X-User-Id"] == "U5000"
    assert captured["headers"]["X-Tenant-Id"] == "tenant-5"
    # The confirmation id doubles as the idempotency key for the Java call.
    assert captured["idempotency_key"] == "b" * 32
    # The business context was reset after the Java call.
    assert get_business_context() == (None, None)


def test_product_refund_order_confirmation_unavailable_mapped_to_ok_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.mcp_servers import product_server

    class _FailingConfirmationStore:
        def require_confirmed(self, confirmation_id: str, *, actor_id: str) -> None:
            raise RedisConnectionError("redis connection refused")

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _FailingConfirmationStore(),
    )
    result = product_server._product_refund_order(
        order_id="A1002",
        reason="七天无理由退货",
        confirmation_id="a" * 32,
        requester_id="user-1",
        user_confirmed=True,
    )
    assert result["ok"] is False
    assert result["error_code"] == "TOOL_CONFIRMATION_UNAVAILABLE"
    assert result["refund"] is None


def test_product_refund_order_success_returns_refund(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    captured: dict[str, object] = {}

    class _FakeOrderClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def refund_order(
            self,
            order_id: str,
            reason: str,
            *,
            idempotency_key: str | None = None,
            trace_context: object = None,
        ) -> dict[str, object]:
            captured["order_id"] = order_id
            captured["idempotency_key"] = idempotency_key
            return {
                "order_id": order_id,
                "order_status": "refunded",
                "payment_status": "refunded",
                "refund_amount": 159.00,
                "refunded_at": "2026-08-06T12:00:00",
                "refund_reason": reason,
            }

    monkeypatch.setattr(
        product_server,
        "JavaOrderClient",
        SimpleNamespace(
            from_settings=staticmethod(
                lambda settings: _FakeOrderClient(settings)
            )
        ),
    )

    class _ConfirmingStore:
        def require_confirmed(
            self,
            confirmation_id: str,
            *,
            actor_id: str,
        ) -> None:
            captured["confirmation_actor_id"] = actor_id
            return None

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _ConfirmingStore(),
    )

    result = product_server._product_refund_order(
        order_id="A1002",
        reason="七天无理由退货",
        confirmation_id="c" * 32,
        requester_id="user-1",
        user_confirmed=True,
    )
    assert result["ok"] is True
    # The confirmation store verified the real requester id, not a demo fallback.
    assert captured["confirmation_actor_id"] == "user-1"
    assert result["allowed"] is True
    assert result["confirmation_checked"] is True
    assert result["confirmation_id"] == "c" * 32
    assert result["error_code"] is None
    assert result["message"] == "退款成功。"
    assert result["refund"]["order_id"] == "A1002"
    assert result["refund"]["payment_status"] == "refunded"
    assert result["refund"]["refund_amount"] == 159.00
    assert captured["idempotency_key"] == "c" * 32


def test_product_refund_order_rejects_missing_requester_id() -> None:
    from app.mcp_servers import product_server

    result = product_server._product_refund_order(
        order_id="A1002",
        reason="七天无理由退货",
        confirmation_id="a" * 32,
        requester_id="",
        user_confirmed=True,
    )
    assert result["ok"] is False
    assert result["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert result["refund"] is None


def test_product_cancel_order_requires_user_confirmation() -> None:
    from app.mcp_servers import product_server

    result = product_server._product_cancel_order(
        order_id="A1002",
        reason="不想要了",
        confirmation_id="a" * 32,
        requester_id="user-1",
        user_confirmed=False,
    )

    assert result["ok"] is False
    assert result["error_code"] == "TOOL_CONFIRMATION_REQUIRED"
    assert result["cancel"] is None


def test_product_cancel_order_validates_confirmation_id_format() -> None:
    server = create_product_mcp_server(_settings())

    async def run() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "cancel_order",
                {
                    "order_id": "A1002",
                    "reason": "不想要了",
                    "requester_id": "user-1",
                    "confirmation_id": "not-a-hex-id",
                },
            )
        # MCP 2.0 surfaces pydantic argument-validation failures as an is_error
        # CallToolResult whose text describes the confirmation_id pattern.
        assert result.is_error is True
        error_text = result.content[0].text
        assert "confirmation_id" in error_text
        assert "string_pattern_mismatch" in error_text

    import asyncio

    asyncio.run(run())


def test_product_cancel_order_sets_business_context_before_java_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    captured: dict[str, object] = {}

    class _ContextCapturingOrderClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def cancel_order(
            self,
            order_id: str,
            reason: str,
            *,
            idempotency_key: str | None = None,
            trace_context: object = None,
        ) -> dict[str, object]:
            captured["order_id"] = order_id
            captured["reason"] = reason
            captured["idempotency_key"] = idempotency_key
            captured["headers"] = build_java_internal_headers(self.settings)
            return {
                "order_id": order_id,
                "order_status": "cancelled",
                "cancel_reason": reason,
            }

    monkeypatch.setattr(
        product_server,
        "JavaOrderClient",
        SimpleNamespace(
            from_settings=staticmethod(
                lambda settings: _ContextCapturingOrderClient(settings)
            )
        ),
    )

    class _ConfirmingStore:
        def require_confirmed(
            self,
            confirmation_id: str,
            *,
            actor_id: str,
        ) -> None:
            captured["confirmation_actor_id"] = actor_id
            return None

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _ConfirmingStore(),
    )

    result = product_server._product_cancel_order(
        order_id="A1002",
        reason="不想要了",
        confirmation_id="b" * 32,
        requester_id="U5000",
        user_confirmed=True,
        user_id="U5000",
        tenant_id="tenant-5",
    )
    assert result["ok"] is True
    # The confirmation store verified the real requester id, not a demo fallback.
    assert captured["confirmation_actor_id"] == "U5000"
    # The Java cancel call saw the injected identity, not the default fallback.
    assert captured["headers"]["X-User-Id"] == "U5000"
    assert captured["headers"]["X-Tenant-Id"] == "tenant-5"
    # The confirmation id doubles as the idempotency key for the Java call.
    assert captured["idempotency_key"] == "b" * 32
    # The business context was reset after the Java call.
    assert get_business_context() == (None, None)


def test_product_cancel_order_confirmation_unavailable_mapped_to_ok_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.mcp_servers import product_server

    class _FailingConfirmationStore:
        def require_confirmed(self, confirmation_id: str, *, actor_id: str) -> None:
            raise RedisConnectionError("redis connection refused")

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _FailingConfirmationStore(),
    )
    result = product_server._product_cancel_order(
        order_id="A1002",
        reason="不想要了",
        confirmation_id="a" * 32,
        requester_id="user-1",
        user_confirmed=True,
    )
    assert result["ok"] is False
    assert result["error_code"] == "TOOL_CONFIRMATION_UNAVAILABLE"
    assert result["cancel"] is None


def test_product_cancel_order_success_returns_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    captured: dict[str, object] = {}

    class _FakeOrderClient:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def cancel_order(
            self,
            order_id: str,
            reason: str,
            *,
            idempotency_key: str | None = None,
            trace_context: object = None,
        ) -> dict[str, object]:
            captured["order_id"] = order_id
            captured["idempotency_key"] = idempotency_key
            return {
                "order_id": order_id,
                "order_status": "cancelled",
                "payment_status": "refunded",
                "cancelled_at": "2026-08-07T10:00:00",
                "cancel_reason": reason,
            }

    monkeypatch.setattr(
        product_server,
        "JavaOrderClient",
        SimpleNamespace(
            from_settings=staticmethod(
                lambda settings: _FakeOrderClient(settings)
            )
        ),
    )

    class _ConfirmingStore:
        def require_confirmed(
            self,
            confirmation_id: str,
            *,
            actor_id: str,
        ) -> None:
            captured["confirmation_actor_id"] = actor_id
            return None

    monkeypatch.setattr(
        product_server,
        "create_tool_confirmation_store",
        lambda: _ConfirmingStore(),
    )

    result = product_server._product_cancel_order(
        order_id="A1002",
        reason="不想要了",
        confirmation_id="c" * 32,
        requester_id="user-1",
        user_confirmed=True,
    )
    assert result["ok"] is True
    # The confirmation store verified the real requester id, not a demo fallback.
    assert captured["confirmation_actor_id"] == "user-1"
    assert result["allowed"] is True
    assert result["confirmation_checked"] is True
    assert result["confirmation_id"] == "c" * 32
    assert result["error_code"] is None
    assert result["message"] == "取消成功。"
    assert result["cancel"]["order_id"] == "A1002"
    assert result["cancel"]["order_status"] == "cancelled"
    assert captured["idempotency_key"] == "c" * 32


def test_product_cancel_order_rejects_missing_requester_id() -> None:
    from app.mcp_servers import product_server

    result = product_server._product_cancel_order(
        order_id="A1002",
        reason="不想要了",
        confirmation_id="a" * 32,
        requester_id="",
        user_confirmed=True,
    )
    assert result["ok"] is False
    assert result["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert result["cancel"] is None


def test_main_refuses_to_start_without_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.mcp_servers import product_server

    monkeypatch.setattr(product_server, "get_settings", lambda: _settings(token=None))
    with pytest.raises(SystemExit) as exc_info:
        product_server.main()
    assert exc_info.value.code == 1


def test_main_starts_with_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.mcp_servers import product_server

    started: dict[str, object] = {}

    def fake_uvicorn_run(app: object, **kwargs: object) -> None:
        started["app"] = app
        started["kwargs"] = kwargs

    monkeypatch.setattr(product_server, "get_settings", lambda: _settings(token="t"))
    monkeypatch.setattr(product_server.uvicorn, "run", fake_uvicorn_run)
    product_server.main()
    assert started["app"] is not None
    assert started["kwargs"]["port"] == 9100
