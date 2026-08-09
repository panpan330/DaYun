import httpx

from app.services.java_error_mapping import (
    TICKET_CONTRACT_REJECTED,
    TICKET_CREATION_UNAVAILABLE,
    build_java_error_app_exception,
    extract_java_error_code,
)


def test_extract_java_error_code_reads_unified_java_error_body() -> None:
    response = httpx.Response(
        403,
        json={
            "success": False,
            "code": "ORDER_ACCESS_DENIED",
            "message": "当前用户无权查看或操作该订单。",
            "data": None,
            "trace_id": "trace-java-001",
        },
    )

    assert extract_java_error_code(response) == "ORDER_ACCESS_DENIED"


def test_build_java_error_app_exception_uses_local_safe_message() -> None:
    response = httpx.Response(
        404,
        json={
            "success": False,
            "code": "ORDER_NOT_FOUND",
            "message": "订单不存在，SQL: select * from orders ...",
        },
    )

    exc = build_java_error_app_exception(
        response,
        operation="order_query",
        fallback_code="TOOL_UPSTREAM_ERROR",
        fallback_message="fallback message",
        fallback_status_code=502,
    )

    assert exc.code == "ORDER_NOT_FOUND"
    assert exc.message == "订单不存在，请确认订单号是否正确。"
    assert "SQL" not in exc.message
    assert exc.status_code == 404


def test_build_java_error_app_exception_hides_internal_auth_failure() -> None:
    response = httpx.Response(
        401,
        json={
            "success": False,
            "code": "INTERNAL_AUTH_FAILED",
            "message": "内部服务鉴权失败。",
        },
    )

    exc = build_java_error_app_exception(
        response,
        operation="ticket_creation",
        fallback_code="TICKET_UPSTREAM_REJECTED",
        fallback_message="fallback message",
        fallback_status_code=502,
    )

    assert exc.code == "TOOL_UPSTREAM_ERROR"
    assert exc.message == TICKET_CREATION_UNAVAILABLE
    assert "鉴权" not in exc.message
    assert exc.status_code == 502


def test_build_java_error_app_exception_maps_idempotency_contract_errors() -> None:
    response = httpx.Response(
        400,
        json={
            "success": False,
            "code": "IDEMPOTENCY_KEY_REQUIRED",
            "message": "写操作缺少幂等键。",
        },
    )

    exc = build_java_error_app_exception(
        response,
        operation="ticket_creation",
        fallback_code="TICKET_UPSTREAM_REJECTED",
        fallback_message="fallback message",
        fallback_status_code=502,
    )

    assert exc.code == "TICKET_UPSTREAM_REJECTED"
    assert exc.message == TICKET_CONTRACT_REJECTED
    assert exc.status_code == 502


def test_build_java_error_app_exception_falls_back_for_unknown_client_error() -> None:
    response = httpx.Response(
        418,
        json={"code": "NEW_JAVA_ERROR"},
    )

    exc = build_java_error_app_exception(
        response,
        operation="order_query",
        fallback_code="TOOL_UPSTREAM_ERROR",
        fallback_message="订单查询服务返回了无法处理的状态，请稍后重试。",
        fallback_status_code=502,
    )

    assert exc.code == "TOOL_UPSTREAM_ERROR"
    assert exc.message == "订单查询服务返回了无法处理的状态，请稍后重试。"
    assert exc.status_code == 502
