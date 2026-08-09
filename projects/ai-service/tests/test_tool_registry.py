import pytest

from app.core.exceptions import AppException
from app.schemas.tool import ToolAccessLevel, ToolDefinition
from app.tools.tool_registry import (
    authorize_tool_call,
    build_openai_chat_tool_definition,
    get_tool_definition,
    list_model_callable_openai_tools,
    list_model_callable_tool_definitions,
    list_tool_definitions,
    require_enabled_tool_definition,
)


def test_get_tool_definition_returns_query_order_definition() -> None:
    definition = get_tool_definition("query_order")

    assert definition is not None
    assert definition.name == "query_order"
    assert definition.access_level == ToolAccessLevel.READ
    assert definition.requires_confirmation is False
    assert definition.enabled is True
    assert set(definition.argument_schema["properties"]) == {"order_id"}


def test_list_tool_definitions_contains_backend_owned_tools() -> None:
    definitions = list_tool_definitions()
    names = {definition.name for definition in definitions}

    assert names == {"query_order", "create_ticket", "refund_order", "cancel_order"}
    assert all(isinstance(definition, ToolDefinition) for definition in definitions)


def test_list_model_callable_tool_definitions_only_exposes_safe_read_tools() -> None:
    definitions = list_model_callable_tool_definitions()

    assert [definition.name for definition in definitions] == ["query_order"]
    assert definitions[0].access_level == ToolAccessLevel.READ
    assert definitions[0].requires_confirmation is False


def test_build_openai_chat_tool_definition_uses_function_tool_shape() -> None:
    definition = get_tool_definition("query_order")
    assert definition is not None

    tool = build_openai_chat_tool_definition(definition)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "query_order"
    assert tool["function"]["parameters"]["type"] == "object"
    assert tool["function"]["parameters"]["additionalProperties"] is False
    assert tool["function"]["strict"] is True


def test_list_model_callable_openai_tools_contains_query_order_schema() -> None:
    tools = list_model_callable_openai_tools()

    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "query_order"
    assert set(tools[0]["function"]["parameters"]["properties"]) == {"order_id"}


def test_authorize_tool_call_allows_read_tool_without_confirmation() -> None:
    definition = authorize_tool_call("query_order")

    assert definition.name == "query_order"
    assert definition.access_level == ToolAccessLevel.READ


def test_authorize_tool_call_rejects_unknown_tool() -> None:
    with pytest.raises(AppException) as exc_info:
        authorize_tool_call("delete_database")

    exc = exc_info.value
    assert exc.code == "TOOL_NOT_ALLOWED"
    assert exc.message == "工具不在允许列表中，后端已拒绝执行。"
    assert exc.status_code == 403


def test_require_enabled_tool_definition_allows_write_tool_before_confirmation() -> None:
    definition = require_enabled_tool_definition("create_ticket")

    assert definition.name == "create_ticket"
    assert definition.requires_confirmation is True


def test_require_enabled_tool_definition_rejects_disabled_tool(
    disabled_sensitive_tool: str,
) -> None:
    with pytest.raises(AppException) as exc_info:
        require_enabled_tool_definition(disabled_sensitive_tool)

    assert exc_info.value.code == "TOOL_NOT_ALLOWED"
    assert exc_info.value.status_code == 403


def test_authorize_tool_call_requires_confirmation_for_write_tool() -> None:
    with pytest.raises(AppException) as exc_info:
        authorize_tool_call("create_ticket")

    exc = exc_info.value
    assert exc.code == "TOOL_CONFIRMATION_REQUIRED"
    assert exc.message == "该工具需要用户确认后才能执行。"
    assert exc.status_code == 409


def test_authorize_tool_call_allows_write_tool_after_confirmation() -> None:
    definition = authorize_tool_call("create_ticket", user_confirmed=True)

    assert definition.name == "create_ticket"
    assert definition.access_level == ToolAccessLevel.WRITE
    assert definition.requires_confirmation is True


def test_authorize_tool_call_rejects_disabled_sensitive_tool_even_when_confirmed(
    disabled_sensitive_tool: str,
) -> None:
    with pytest.raises(AppException) as exc_info:
        authorize_tool_call(disabled_sensitive_tool, user_confirmed=True)

    exc = exc_info.value
    assert exc.code == "TOOL_NOT_ALLOWED"
    assert exc.status_code == 403


def test_authorize_tool_call_allows_refund_order_after_confirmation() -> None:
    definition = authorize_tool_call("refund_order", user_confirmed=True)

    assert definition.name == "refund_order"
    assert definition.access_level == ToolAccessLevel.SENSITIVE
    assert definition.requires_confirmation is True


def test_refund_order_tool_is_enabled_and_requires_confirmation() -> None:
    definition = get_tool_definition("refund_order")

    assert definition is not None
    assert definition.enabled is True
    assert definition.access_level == ToolAccessLevel.SENSITIVE
    assert definition.requires_confirmation is True
    assert definition.argument_schema["type"] == "object"
    assert set(definition.argument_schema["properties"]) == {
        "order_id",
        "reason",
        "requester_id",
    }
    assert set(definition.argument_schema["required"]) == {
        "order_id",
        "reason",
        "requester_id",
    }


def test_refund_order_not_in_read_only_model_callable_tools() -> None:
    definitions = list_model_callable_tool_definitions()

    assert "refund_order" not in {definition.name for definition in definitions}


def test_cancel_order_tool_is_enabled_and_requires_confirmation() -> None:
    definition = get_tool_definition("cancel_order")

    assert definition is not None
    assert definition.enabled is True
    assert definition.access_level == ToolAccessLevel.SENSITIVE
    assert definition.requires_confirmation is True
    assert definition.argument_schema["type"] == "object"
    assert set(definition.argument_schema["properties"]) == {
        "order_id",
        "reason",
        "requester_id",
    }
    assert set(definition.argument_schema["required"]) == {
        "order_id",
        "reason",
        "requester_id",
    }
