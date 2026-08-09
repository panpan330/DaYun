import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.exceptions import AppException
from app.services.java_business_contract import (
    JavaOrderToolView,
    JavaTicketToolView,
    validate_java_success_envelope,
)
from app.services.java_error_mapping import build_java_error_app_exception


CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "contracts"
    / "java-business-service"
    / "internal-api-contract-cases.json"
)


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def contract_case(case_id: str) -> dict[str, Any]:
    for item in load_contract()["cases"]:
        if item["id"] == case_id:
            return item
    raise AssertionError(f"missing contract case: {case_id}")


def test_shared_contract_defines_core_python_java_cases() -> None:
    ids = {item["id"] for item in load_contract()["cases"]}

    assert {
        "query_order_success",
        "query_order_access_denied",
        "create_ticket_success",
        "create_ticket_missing_idempotency_key",
    }.issubset(ids)


def test_python_accepts_java_order_success_envelope_from_contract() -> None:
    case = contract_case("query_order_success")
    trace_id = load_contract()["common_headers"]["X-Trace-Id"]
    payload = {
        "success": case["expected"]["success"],
        "code": case["expected"]["code"],
        "message": "OK",
        "data": {
            "order_id": "A1001",
            "order_status": "shipped",
            "payment_status": "paid",
            "logistics_message": "订单已发货，正在运输中。",
            "latest_event": "包裹已离开发货仓。",
            "can_create_ticket": True,
            "user_visible_summary": "订单已发货，正在运输中。",
        },
        "trace_id": trace_id,
    }

    view = validate_java_success_envelope(payload, data_model=JavaOrderToolView)

    assert view.order_id == "A1001"
    assert view.can_create_ticket is True


def test_python_accepts_java_ticket_success_envelope_from_contract() -> None:
    case = contract_case("create_ticket_success")
    trace_id = load_contract()["common_headers"]["X-Trace-Id"]
    payload = {
        "success": case["expected"]["success"],
        "code": case["expected"]["code"],
        "message": "OK",
        "data": {
            "ticket_id": "T-123e4567-e89b-12d3-a456-426614174000",
            "ticket_status": "created",
            "title": case["body"]["title"],
            "category": case["body"]["category"],
            "priority": case["body"]["priority"],
            "related_order_id": case["body"]["related_order_id"],
            "created_at": "2026-07-28T09:00:00Z",
            "user_visible_summary": "工单已创建，客服会继续跟进。",
        },
        "trace_id": trace_id,
    }

    view = validate_java_success_envelope(payload, data_model=JavaTicketToolView)

    assert view.ticket_id.startswith("T-")
    assert view.related_order_id == "A1001"


@pytest.mark.parametrize(
    "case_id",
    ["query_order_access_denied", "create_ticket_missing_idempotency_key"],
)
def test_python_error_mapping_matches_shared_contract(case_id: str) -> None:
    case = contract_case(case_id)
    expected = case["expected"]
    response = httpx.Response(
        expected["status"],
        json={
            "success": expected["success"],
            "code": expected["code"],
            "message": "Java provider message should not decide Python user wording.",
            "data": None,
            "trace_id": load_contract()["common_headers"]["X-Trace-Id"],
        },
    )

    exc = build_java_error_app_exception(
        response,
        operation=case["operation"],
        fallback_code="TICKET_UPSTREAM_REJECTED",
        fallback_message="fallback",
        fallback_status_code=502,
    )

    assert exc.code == expected["python_error_code"]
    assert exc.status_code == expected["python_status"]


def test_python_rejects_java_success_envelope_missing_contract_field() -> None:
    payload = {
        "success": True,
        "code": "OK",
        "message": "OK",
        "data": {
            "order_id": "A1001",
            "order_status": "shipped",
            "payment_status": "paid",
            "logistics_message": "订单已发货，正在运输中。",
            "can_create_ticket": True,
            "user_visible_summary": "订单已发货，正在运输中。",
        },
        "trace_id": "trace-contract-stage7-11",
    }

    with pytest.raises(AppException) as exc_info:
        validate_java_success_envelope(payload, data_model=JavaOrderToolView)

    assert exc_info.value.code == "JAVA_CONTRACT_VALIDATION_FAILED"
