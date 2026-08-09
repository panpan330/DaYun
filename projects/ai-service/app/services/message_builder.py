from collections.abc import Sequence

from app.schemas.chat import ChatMessage, ChatMessageRole


DEFAULT_SYSTEM_MESSAGE = "你是一个耐心的编程学习助手，回答要简洁清楚。"


def serialize_chat_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [message.to_openai_dict() for message in messages]


def build_single_turn_messages(
    user_message: str,
    *,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> list[ChatMessage]:
    return [
        ChatMessage(role=ChatMessageRole.SYSTEM, content=system_message),
        ChatMessage(role=ChatMessageRole.USER, content=user_message),
    ]


def build_multi_turn_messages(
    user_message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> list[ChatMessage]:
    messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_message)]
    if history:
        messages.extend(history)
    messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_message))
    return messages
