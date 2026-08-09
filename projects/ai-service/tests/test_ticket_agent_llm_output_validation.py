import json

import pytest

from app.agents.ticket_agent import (
    get_ticket_field_extraction_json_schema,
    get_ticket_intent_classification_json_schema,
    parse_ticket_field_extraction_json,
    parse_ticket_intent_classification_json,
)
from app.core.exceptions import AppException


def make_valid_field_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "issue_type": "complaint",
        "order_id": "A2001",
        "description": "商品破损，用户希望客服处理订单 A2001。",
        "user_request": "人工处理商品破损投诉",
        "urgency": "high",
        "need_human_review": True,
    }
    payload.update(overrides)
    return payload


def assert_validation_failed(exc_info: pytest.ExceptionInfo[AppException]) -> None:
    assert exc_info.value.status_code == 502
    assert exc_info.value.details


def test_intent_output_json_schema_forbids_extra_properties() -> None:
    schema = get_ticket_intent_classification_json_schema()

    assert schema["additionalProperties"] is False


def test_field_output_json_schema_forbids_extra_properties_and_limits_order_id() -> None:
    schema = get_ticket_field_extraction_json_schema()
    order_id_schema = schema["properties"]["order_id"]
    order_id_branches = [order_id_schema, *order_id_schema.get("anyOf", [])]

    assert schema["additionalProperties"] is False
    assert any(
        branch.get("pattern") == "^[A-Za-z0-9_-]+$"
        for branch in order_id_branches
    )


def test_intent_output_rejects_extra_model_field() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_intent_classification_json(
            json.dumps(
                {
                    "intent": "ticket_request",
                    "reason": "用户要求人工处理。",
                    "confidence": 0.91,
                },
                ensure_ascii=False,
            )
        )

    assert exc_info.value.code == "TICKET_INTENT_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)


def test_intent_output_rejects_blank_reason_after_normalization() -> None:
    with pytest.raises(AppException) as exc_info:
        parse_ticket_intent_classification_json(
            json.dumps(
                {
                    "intent": "ticket_request",
                    "reason": "   ",
                },
                ensure_ascii=False,
            )
        )

    assert exc_info.value.code == "TICKET_INTENT_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)


def test_field_output_rejects_should_create_ticket_extra_field() -> None:
    payload = make_valid_field_payload(should_create_ticket=True)

    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json(
            json.dumps(payload, ensure_ascii=False)
        )

    assert exc_info.value.code == "TICKET_FIELD_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)


def test_field_output_rejects_invalid_order_id_text() -> None:
    payload = make_valid_field_payload(order_id="我猜是 A2001")

    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json(
            json.dumps(payload, ensure_ascii=False)
        )

    assert exc_info.value.code == "TICKET_FIELD_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)


@pytest.mark.parametrize("order_id", ["", "  ", "null", "None", "未提供", "未知"])
def test_field_output_normalizes_null_like_order_id_to_none(order_id: str) -> None:
    payload = make_valid_field_payload(
        issue_type="policy_gap",
        order_id=order_id,
        urgency="normal",
    )

    fields = parse_ticket_field_extraction_json(
        json.dumps(payload, ensure_ascii=False)
    )

    assert fields["order_id"] is None


def test_field_output_rejects_string_boolean_for_human_review() -> None:
    payload = make_valid_field_payload(need_human_review="true")

    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json(
            json.dumps(payload, ensure_ascii=False)
        )

    assert exc_info.value.code == "TICKET_FIELD_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)


def test_field_output_rejects_blank_description_after_normalization() -> None:
    payload = make_valid_field_payload(description="   ")

    with pytest.raises(AppException) as exc_info:
        parse_ticket_field_extraction_json(
            json.dumps(payload, ensure_ascii=False)
        )

    assert exc_info.value.code == "TICKET_FIELD_LLM_VALIDATION_FAILED"
    assert_validation_failed(exc_info)
