from app.core.exceptions import AppException
from app.schemas.cancel import get_cancel_order_args_json_schema
from app.schemas.refund import get_refund_order_args_json_schema
from app.schemas.tool import (
    ToolAccessLevel,
    ToolDefinition,
    get_query_order_args_json_schema,
)
from app.schemas.ticket import get_create_ticket_args_json_schema


REFUND_ORDER_TOOL_NAME = "refund_order"
CANCEL_ORDER_TOOL_NAME = "cancel_order"


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "query_order": ToolDefinition(
        name="query_order",
        description="查询订单状态和物流摘要，只读取订单信息，不修改业务数据。",
        access_level=ToolAccessLevel.READ,
        requires_confirmation=False,
        enabled=True,
        argument_schema=get_query_order_args_json_schema(),
    ),
    "create_ticket": ToolDefinition(
        name="create_ticket",
        description="创建客服工单，会写入业务系统，必须先让用户确认。",
        access_level=ToolAccessLevel.WRITE,
        requires_confirmation=True,
        enabled=True,
        argument_schema=get_create_ticket_args_json_schema(),
    ),
    REFUND_ORDER_TOOL_NAME: ToolDefinition(
        name=REFUND_ORDER_TOOL_NAME,
        description="发起退款操作，属于敏感业务动作，必须先让用户确认，且订单未发货才可退。",
        access_level=ToolAccessLevel.SENSITIVE,
        requires_confirmation=True,
        enabled=True,
        argument_schema=get_refund_order_args_json_schema(),
    ),
    CANCEL_ORDER_TOOL_NAME: ToolDefinition(
        name=CANCEL_ORDER_TOOL_NAME,
        description="发起取消订单操作，属于敏感业务动作，必须先让用户确认，且订单未发货才可取消。",
        access_level=ToolAccessLevel.SENSITIVE,
        requires_confirmation=True,
        enabled=True,
        argument_schema=get_cancel_order_args_json_schema(),
    ),
}


def get_tool_definition(tool_name: str) -> ToolDefinition | None:
    return TOOL_REGISTRY.get(tool_name)


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY.values())


def list_model_callable_tool_definitions() -> list[ToolDefinition]:
    return [
        definition
        for definition in list_tool_definitions()
        if definition.enabled
        and definition.access_level == ToolAccessLevel.READ
        and not definition.requires_confirmation
    ]


def build_openai_chat_tool_definition(
    definition: ToolDefinition,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.argument_schema,
            "strict": True,
        },
    }


def list_model_callable_openai_tools(
    tool_names: list[str] | None = None,
) -> list[dict[str, object]]:
    """构造暴露给模型的 OpenAI tools；tool_names=None → 全量（向后兼容）。"""
    definitions = list_model_callable_tool_definitions()
    if tool_names is not None:
        allowed = set(tool_names)
        definitions = [d for d in definitions if d.name in allowed]
    return [
        build_openai_chat_tool_definition(definition)
        for definition in definitions
    ]


def require_enabled_tool_definition(tool_name: str) -> ToolDefinition:
    definition = get_tool_definition(tool_name)
    if definition is None or not definition.enabled:
        raise AppException(
            code="TOOL_NOT_ALLOWED",
            message="工具不在允许列表中，后端已拒绝执行。",
            status_code=403,
        )
    return definition


def authorize_tool_call(
    tool_name: str,
    *,
    user_confirmed: bool = False,
) -> ToolDefinition:
    definition = require_enabled_tool_definition(tool_name)

    if definition.requires_confirmation and not user_confirmed:
        raise AppException(
            code="TOOL_CONFIRMATION_REQUIRED",
            message="该工具需要用户确认后才能执行。",
            status_code=409,
        )

    return definition
