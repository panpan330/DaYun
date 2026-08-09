from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage, ChatMessageRole
from app.services.langchain_chat_model_service import (
    LangChainChatModelService,
    build_langchain_chat_messages,
    create_langchain_chat_model,
    extract_langchain_reply,
)


class FakeLangChainChatModel:
    def __init__(
        self,
        response: Any | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response or AIMessage(content="  LangChain 模型回复  ")
        self.error = error
        self.calls: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def test_create_langchain_chat_model_uses_project_settings() -> None:
    model = create_langchain_chat_model(
        Settings(
            llm_api_key="test-key",
            llm_model="qwen-test",
            llm_base_url="https://example.com/v1/",
            request_timeout_seconds=3,
            llm_max_retries=1,
            _env_file=None,
        )
    )

    assert model.model_name == "qwen-test"
    assert str(model.openai_api_base).rstrip("/") == "https://example.com/v1"
    assert model.request_timeout == 3.0
    assert model.max_retries == 1


def test_create_langchain_chat_model_requires_api_key() -> None:
    with pytest.raises(ValueError, match="LLM_API_KEY is not configured"):
        create_langchain_chat_model(Settings(_env_file=None))


def test_build_langchain_chat_messages_converts_project_messages() -> None:
    history = [
        ChatMessage(role=ChatMessageRole.USER, content="什么是 API？"),
        ChatMessage(role=ChatMessageRole.ASSISTANT, content="API 是程序之间的接口。"),
    ]

    messages = build_langchain_chat_messages("那 FastAPI 呢？", history=history)

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert messages[1].content == "什么是 API？"
    assert messages[2].content == "API 是程序之间的接口。"
    assert "## 任务\n那 FastAPI 呢？" in str(messages[3].content)


def test_extract_langchain_reply_reads_ai_message_text() -> None:
    assert extract_langchain_reply(AIMessage(content="  你好  ")) == "你好"


def test_extract_langchain_reply_rejects_empty_content() -> None:
    with pytest.raises(AppException) as exc_info:
        extract_langchain_reply(AIMessage(content="   "))

    assert exc_info.value.code == "LLM_EMPTY_RESPONSE"
    assert exc_info.value.status_code == 502


def test_langchain_chat_model_service_invokes_model_with_messages() -> None:
    fake_model = FakeLangChainChatModel()
    service = LangChainChatModelService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=fake_model,
    )

    reply = service.generate_reply("解释 LangChain ChatModel")

    assert reply == "LangChain 模型回复"
    assert len(fake_model.calls) == 1
    messages = fake_model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "## 任务\n解释 LangChain ChatModel" in str(messages[1].content)


def test_langchain_chat_model_service_passes_history_to_model() -> None:
    fake_model = FakeLangChainChatModel(AIMessage(content="FastAPI 是 Web 框架。"))
    service = LangChainChatModelService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=fake_model,
    )
    history = [
        ChatMessage(role=ChatMessageRole.USER, content="什么是 API？"),
        ChatMessage(role=ChatMessageRole.ASSISTANT, content="API 是程序之间的接口。"),
    ]

    reply = service.generate_reply("那 FastAPI 呢？", history=history)

    assert reply == "FastAPI 是 Web 框架。"
    messages = fake_model.calls[0]
    assert [type(message) for message in messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert messages[1].content == "什么是 API？"
    assert messages[2].content == "API 是程序之间的接口。"


def test_langchain_chat_model_service_requires_api_key() -> None:
    service = LangChainChatModelService(Settings(_env_file=None), model=FakeLangChainChatModel())

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("解释 LangChain")

    assert exc_info.value.code == "LLM_API_KEY_MISSING"
    assert exc_info.value.status_code == 500


def test_langchain_chat_model_service_maps_unknown_model_error() -> None:
    service = LangChainChatModelService(
        Settings(llm_api_key="test-key", llm_model="qwen-test", _env_file=None),
        model=FakeLangChainChatModel(error=RuntimeError("provider failed")),
    )

    with pytest.raises(AppException) as exc_info:
        service.generate_reply("解释 LangChain")

    assert exc_info.value.code == "LLM_CALL_FAILED"
    assert exc_info.value.status_code == 502
