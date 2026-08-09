import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import AppException
from app.core.trace import TRACE_ID_HEADER
from app.schemas.tool import (
    OrderStatus,
    PaymentStatus,
    QueryOrderArgs,
    QueryOrderResult,
)
from app.tools.idempotency import IDEMPOTENCY_KEY_HEADER


def make_query_order_result(order_id: str = "A1001") -> QueryOrderResult:
    return QueryOrderResult(
        order_id=order_id,
        order_status=OrderStatus.WAITING_SHIPMENT,
        payment_status=PaymentStatus.PAID,
        logistics_message="商家已接单，等待仓库发货。",
        latest_event="仓库正在准备出库。",
        can_create_ticket=True,
        source="java_mock_service",
    )


def patch_query_order_tool(
    monkeypatch: pytest.MonkeyPatch,
    *,
    call_count_ref: dict[str, int] | None = None,
) -> None:
    from app.routers import tools as tools_router

    def fake_query_order_tool(
        arguments: QueryOrderArgs,
        **kwargs: object,
    ) -> QueryOrderResult:
        if call_count_ref is not None:
            call_count_ref["count"] += 1

        if arguments.order_id == "A9999":
            raise AppException(
                code="ORDER_NOT_FOUND",
                message="订单不存在，请确认订单号是否正确。",
                status_code=404,
            )
        if arguments.order_id == "A_TIMEOUT":
            raise AppException(
                code="TOOL_TIMEOUT",
                message="订单查询工具调用超时，请稍后重试。",
                status_code=504,
            )
        if arguments.order_id == "A500":
            raise AppException(
                code="TOOL_UPSTREAM_ERROR",
                message="订单查询服务暂时不可用，请稍后重试。",
                status_code=502,
            )

        return make_query_order_result(arguments.order_id)

    monkeypatch.setattr(
        tools_router,
        "run_query_order_tool",
        fake_query_order_tool,
    )


def test_query_order_api_returns_java_mock_order(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-query-order"},
        json={"order_id": "A1001"},
    )
    data = response.json()

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "trace-query-order"
    assert data == {
        "result": {
            "order_id": "A1001",
            "order_status": "waiting_shipment",
            "payment_status": "paid",
            "logistics_message": "商家已接单，等待仓库发货。",
            "latest_event": "仓库正在准备出库。",
            "can_create_ticket": True,
            "source": "java_mock_service",
        }
    }


def test_list_langchain_tools_returns_query_order_metadata(
    client: TestClient,
) -> None:
    response = client.get(
        "/tools/langchain",
        headers={TRACE_ID_HEADER: "trace-langchain-tools"},
    )
    data = response.json()

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "trace-langchain-tools"
    assert data["tools"][0]["name"] == "query_order"
    assert data["tools"][0]["description"] == "查询订单状态和物流摘要，只读取订单信息，不修改业务数据。"
    assert "order_id" in data["tools"][0]["args_schema"]


def test_langchain_query_order_api_invokes_langchain_tool(
    app: object,
    client: TestClient,
) -> None:
    from app.routers.tools import get_query_order_langchain_tool

    class FakeLangChainQueryOrderTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def invoke(self, arguments: dict[str, str]) -> dict[str, object]:
            self.calls.append(arguments)
            return make_query_order_result(arguments["order_id"]).model_dump(mode="json")

    fake_tool = FakeLangChainQueryOrderTool()
    app.dependency_overrides[get_query_order_langchain_tool] = lambda: fake_tool

    response = client.post(
        "/tools/langchain/query-order",
        headers={TRACE_ID_HEADER: "trace-langchain-query-order"},
        json={"order_id": "A1001"},
    )

    assert response.status_code == 200
    assert response.headers[TRACE_ID_HEADER] == "trace-langchain-query-order"
    assert response.json()["result"]["order_id"] == "A1001"
    assert response.json()["result"]["source"] == "java_mock_service"
    assert fake_tool.calls == [{"order_id": "A1001"}]


def test_langchain_query_order_api_rejects_invalid_arguments(
    client: TestClient,
) -> None:
    response = client.post(
        "/tools/langchain/query-order",
        headers={TRACE_ID_HEADER: "trace-langchain-query-order-invalid"},
        json={"order_id": "A 1001"},
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["trace_id"] == "trace-langchain-query-order-invalid"
    assert data["details"][0]["loc"] == ["body", "order_id"]


def test_query_order_api_reuses_result_for_same_idempotency_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count_ref = {"count": 0}
    patch_query_order_tool(monkeypatch, call_count_ref=call_count_ref)

    first_response = client.post(
        "/tools/query-order",
        headers={
            TRACE_ID_HEADER: "trace-query-order-idem-first",
            IDEMPOTENCY_KEY_HEADER: "query-order-api-key-001",
        },
        json={"order_id": "A1001"},
    )
    second_response = client.post(
        "/tools/query-order",
        headers={
            TRACE_ID_HEADER: "trace-query-order-idem-second",
            IDEMPOTENCY_KEY_HEADER: "query-order-api-key-001",
        },
        json={"order_id": "A1001"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()
    assert call_count_ref["count"] == 1


def test_query_order_api_rejects_same_idempotency_key_with_different_body(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    first_response = client.post(
        "/tools/query-order",
        headers={
            TRACE_ID_HEADER: "trace-query-order-idem-conflict-first",
            IDEMPOTENCY_KEY_HEADER: "query-order-api-key-002",
        },
        json={"order_id": "A1001"},
    )
    second_response = client.post(
        "/tools/query-order",
        headers={
            TRACE_ID_HEADER: "trace-query-order-idem-conflict-second",
            IDEMPOTENCY_KEY_HEADER: "query-order-api-key-002",
        },
        json={"order_id": "A1002"},
    )
    data = second_response.json()

    assert first_response.status_code == 200
    assert second_response.status_code == 409
    assert data == {
        "code": "IDEMPOTENCY_KEY_CONFLICT",
        "message": "同一个幂等键不能用于不同的工具调用参数。",
        "trace_id": "trace-query-order-idem-conflict-second",
    }


def test_query_order_api_rejects_invalid_idempotency_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    response = client.post(
        "/tools/query-order",
        headers={
            TRACE_ID_HEADER: "trace-query-order-idem-invalid",
            IDEMPOTENCY_KEY_HEADER: "abc",
        },
        json={"order_id": "A1001"},
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "IDEMPOTENCY_KEY_INVALID"
    assert data["trace_id"] == "trace-query-order-idem-invalid"


def test_query_order_api_returns_not_found_for_missing_order(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-missing"},
        json={"order_id": "A9999"},
    )
    data = response.json()

    assert response.status_code == 404
    assert data == {
        "code": "ORDER_NOT_FOUND",
        "message": "订单不存在，请确认订单号是否正确。",
        "trace_id": "trace-order-missing",
    }


def test_query_order_api_returns_timeout_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-timeout"},
        json={"order_id": "A_TIMEOUT"},
    )
    data = response.json()

    assert response.status_code == 504
    assert data == {
        "code": "TOOL_TIMEOUT",
        "message": "订单查询工具调用超时，请稍后重试。",
        "trace_id": "trace-order-timeout",
    }


def test_query_order_api_returns_upstream_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_query_order_tool(monkeypatch)

    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-upstream"},
        json={"order_id": "A500"},
    )
    data = response.json()

    assert response.status_code == 502
    assert data == {
        "code": "TOOL_UPSTREAM_ERROR",
        "message": "订单查询服务暂时不可用，请稍后重试。",
        "trace_id": "trace-order-upstream",
    }


def test_query_order_api_rejects_missing_order_id(client: TestClient) -> None:
    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-id-missing"},
        json={},
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["message"] == "请求参数校验失败"
    assert data["trace_id"] == "trace-order-id-missing"
    assert data["details"][0]["loc"] == ["body", "order_id"]
    assert data["details"][0]["type"] == "missing"


def test_query_order_api_rejects_invalid_order_id(client: TestClient) -> None:
    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-id-invalid"},
        json={"order_id": "A 1001"},
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["trace_id"] == "trace-order-id-invalid"
    assert data["details"][0]["loc"] == ["body", "order_id"]
    assert data["details"][0]["type"] == "string_pattern_mismatch"


def test_query_order_api_rejects_extra_request_fields(client: TestClient) -> None:
    response = client.post(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-order-extra"},
        json={"order_id": "A1001", "admin": True},
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["trace_id"] == "trace-order-extra"
    assert data["details"][0]["loc"] == ["body", "admin"]
    assert data["details"][0]["type"] == "extra_forbidden"


def test_query_order_api_does_not_allow_get(client: TestClient) -> None:
    response = client.get(
        "/tools/query-order",
        headers={TRACE_ID_HEADER: "trace-query-order-method"},
    )
    data = response.json()

    assert response.status_code == 405
    assert data == {
        "code": "METHOD_NOT_ALLOWED",
        "message": "请求方法不允许",
        "trace_id": "trace-query-order-method",
    }


def test_tool_confirmation_api_creates_then_confirms_exact_plan(
    client: TestClient,
) -> None:
    create_response = client.post(
        "/tools/confirmations",
        headers={TRACE_ID_HEADER: "trace-confirmation-create"},
        json={
            "actor_id": "demo_user_001",
            "tool_name": "create_ticket",
            "arguments": {
                "title": "订单 A1001 未发货",
                "description": "用户反馈订单迟迟未发货。",
                "order_id": "A1001",
            },
        },
    )
    pending = create_response.json()

    assert create_response.status_code == 200
    assert create_response.headers[TRACE_ID_HEADER] == "trace-confirmation-create"
    assert pending["status"] == "pending"
    assert pending["tool_name"] == "create_ticket"
    assert pending["arguments"]["order_id"] == "A1001"
    assert len(pending["confirmation_id"]) == 32
    assert "不会执行" in pending["message"]

    confirm_response = client.post(
        f"/tools/confirmations/{pending['confirmation_id']}/confirm",
        headers={TRACE_ID_HEADER: "trace-confirmation-confirm"},
        json={"actor_id": "demo_user_001"},
    )
    confirmed = confirm_response.json()

    assert confirm_response.status_code == 200
    assert confirm_response.headers[TRACE_ID_HEADER] == "trace-confirmation-confirm"
    assert confirmed["status"] == "confirmed"
    assert confirmed["tool_name"] == "create_ticket"
    assert confirmed["arguments"] == pending["arguments"]
    assert confirmed["arguments_fingerprint"] == pending["arguments_fingerprint"]
    assert "专用执行接口" in confirmed["message"]


def test_tool_confirmation_api_rejects_other_actor(client: TestClient) -> None:
    create_response = client.post(
        "/tools/confirmations",
        json={
            "actor_id": "demo_user_001",
            "tool_name": "create_ticket",
            "arguments": {"title": "订单 A1001 未发货"},
        },
    )
    confirmation_id = create_response.json()["confirmation_id"]

    response = client.post(
        f"/tools/confirmations/{confirmation_id}/confirm",
        headers={TRACE_ID_HEADER: "trace-confirmation-other-actor"},
        json={"actor_id": "other_user_002"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "code": "TOOL_CONFIRMATION_FORBIDDEN",
        "message": "当前操作者不能确认其他人的工具请求。",
        "trace_id": "trace-confirmation-other-actor",
    }


def test_tool_confirmation_api_rejects_read_tool(client: TestClient) -> None:
    response = client.post(
        "/tools/confirmations",
        headers={TRACE_ID_HEADER: "trace-confirmation-read-tool"},
        json={
            "actor_id": "demo_user_001",
            "tool_name": "query_order",
            "arguments": {"order_id": "A1001"},
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "code": "TOOL_CONFIRMATION_NOT_REQUIRED",
        "message": "该工具不需要用户确认。",
        "trace_id": "trace-confirmation-read-tool",
    }


def test_tool_confirmation_api_rejects_arguments_change_in_confirm_request(
    client: TestClient,
) -> None:
    create_response = client.post(
        "/tools/confirmations",
        json={
            "actor_id": "demo_user_001",
            "tool_name": "create_ticket",
            "arguments": {"title": "订单 A1001 未发货"},
        },
    )
    confirmation_id = create_response.json()["confirmation_id"]

    response = client.post(
        f"/tools/confirmations/{confirmation_id}/confirm",
        headers={TRACE_ID_HEADER: "trace-confirmation-arguments-change"},
        json={
            "actor_id": "demo_user_001",
            "arguments": {"title": "改成退款"},
        },
    )
    data = response.json()

    assert response.status_code == 422
    assert data["code"] == "VALIDATION_ERROR"
    assert data["trace_id"] == "trace-confirmation-arguments-change"
    assert data["details"][0]["loc"] == ["body", "arguments"]
    assert data["details"][0]["type"] == "extra_forbidden"


def test_tool_confirmation_api_does_not_allow_get(client: TestClient) -> None:
    response = client.get(
        "/tools/confirmations",
        headers={TRACE_ID_HEADER: "trace-confirmation-method"},
    )

    assert response.status_code == 405
    assert response.json() == {
        "code": "METHOD_NOT_ALLOWED",
        "message": "请求方法不允许",
        "trace_id": "trace-confirmation-method",
    }
