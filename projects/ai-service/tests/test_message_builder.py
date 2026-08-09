from app.schemas.chat import ChatMessage, ChatMessageRole
from app.services.message_builder import (
    DEFAULT_SYSTEM_MESSAGE,
    build_multi_turn_messages,
    build_single_turn_messages,
    serialize_chat_messages,
)


def test_build_single_turn_messages_contains_system_and_user_messages() -> None:
    messages = build_single_turn_messages("请解释 FastAPI 是什么")

    assert messages == [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=DEFAULT_SYSTEM_MESSAGE),
        ChatMessage(role=ChatMessageRole.USER, content="请解释 FastAPI 是什么"),
    ]


def test_build_single_turn_messages_accepts_custom_system_message() -> None:
    messages = build_single_turn_messages(
        "请解释 FastAPI 是什么",
        system_message="你是一个只用一句话回答的助手。",
    )

    assert messages[0] == ChatMessage(
        role=ChatMessageRole.SYSTEM,
        content="你是一个只用一句话回答的助手。",
    )


def test_build_multi_turn_messages_keeps_history_order() -> None:
    history = [
        ChatMessage(role=ChatMessageRole.USER, content="什么是 API？"),
        ChatMessage(role=ChatMessageRole.ASSISTANT, content="API 是程序之间的接口。"),
    ]

    messages = build_multi_turn_messages("那 FastAPI 呢？", history=history)

    assert messages == [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=DEFAULT_SYSTEM_MESSAGE),
        ChatMessage(role=ChatMessageRole.USER, content="什么是 API？"),
        ChatMessage(role=ChatMessageRole.ASSISTANT, content="API 是程序之间的接口。"),
        ChatMessage(role=ChatMessageRole.USER, content="那 FastAPI 呢？"),
    ]


def test_serialize_chat_messages_returns_sdk_friendly_dicts() -> None:
    messages = [
        ChatMessage(role=ChatMessageRole.SYSTEM, content="你是一个老师。"),
        ChatMessage(role=ChatMessageRole.USER, content="解释 token。"),
    ]

    assert serialize_chat_messages(messages) == [
        {"role": "system", "content": "你是一个老师。"},
        {"role": "user", "content": "解释 token。"},
    ]
