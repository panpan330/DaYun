import pytest

from app.core.exceptions import AppException
from app.schemas.tool import QueryOrderArgs
from app.tools.idempotency import (
    build_arguments_fingerprint,
    get_idempotency_record_count,
    run_idempotent_tool,
    validate_idempotency_key,
)


def test_validate_idempotency_key_accepts_supported_key() -> None:
    assert validate_idempotency_key(" order-query-001 ") == "order-query-001"


def test_validate_idempotency_key_treats_missing_or_blank_key_as_disabled() -> None:
    assert validate_idempotency_key(None) is None
    assert validate_idempotency_key("   ") is None


def test_validate_idempotency_key_rejects_unsafe_key() -> None:
    with pytest.raises(AppException) as exc_info:
        validate_idempotency_key("abc")

    exc = exc_info.value
    assert exc.code == "IDEMPOTENCY_KEY_INVALID"
    assert exc.status_code == 422


def test_build_arguments_fingerprint_is_stable_for_same_arguments() -> None:
    first = build_arguments_fingerprint(
        "query_order",
        {"order_id": "A1001", "source": "test"},
    )
    second = build_arguments_fingerprint(
        "query_order",
        {"source": "test", "order_id": "A1001"},
    )

    assert first == second


def test_build_arguments_fingerprint_changes_when_arguments_change() -> None:
    first = build_arguments_fingerprint("query_order", {"order_id": "A1001"})
    second = build_arguments_fingerprint("query_order", {"order_id": "A1002"})

    assert first != second


def test_run_idempotent_tool_without_key_executes_every_time() -> None:
    call_count = 0

    def executor() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {"call_count": call_count}

    first_result = run_idempotent_tool(
        "query_order",
        QueryOrderArgs(order_id="A1001"),
        None,
        executor,
    )
    second_result = run_idempotent_tool(
        "query_order",
        QueryOrderArgs(order_id="A1001"),
        None,
        executor,
    )

    assert first_result == {"call_count": 1}
    assert second_result == {"call_count": 2}
    assert call_count == 2
    assert get_idempotency_record_count() == 0


def test_run_idempotent_tool_with_same_key_reuses_first_result() -> None:
    call_count = 0

    def executor() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        return {"call_count": call_count}

    first_result = run_idempotent_tool(
        "query_order",
        QueryOrderArgs(order_id="A1001"),
        "query-order-key-001",
        executor,
    )
    second_result = run_idempotent_tool(
        "query_order",
        QueryOrderArgs(order_id="A1001"),
        "query-order-key-001",
        executor,
    )

    assert first_result == {"call_count": 1}
    assert second_result == {"call_count": 1}
    assert first_result is not second_result
    assert call_count == 1
    assert get_idempotency_record_count() == 1


def test_run_idempotent_tool_rejects_same_key_with_different_arguments() -> None:
    run_idempotent_tool(
        "query_order",
        QueryOrderArgs(order_id="A1001"),
        "query-order-key-002",
        lambda: {"order_id": "A1001"},
    )

    with pytest.raises(AppException) as exc_info:
        run_idempotent_tool(
            "query_order",
            QueryOrderArgs(order_id="A1002"),
            "query-order-key-002",
            lambda: {"order_id": "A1002"},
        )

    exc = exc_info.value
    assert exc.code == "IDEMPOTENCY_KEY_CONFLICT"
    assert exc.status_code == 409
