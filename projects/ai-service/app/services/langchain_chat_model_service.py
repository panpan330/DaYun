import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.chat import ChatMessage, ChatMessageRole
from app.services.llm_service import build_chat_messages, map_openai_error_to_app_exception


logger = logging.getLogger(__name__)


def create_langchain_chat_model(settings: Settings) -> ChatOpenAI:
    api_key = settings.resolved_llm_api_key
    if api_key is None:
        raise ValueError("LLM_API_KEY is not configured")

    model_kwargs: dict[str, object] = {
        "model": settings.llm_model,
        "api_key": api_key,
        "timeout": settings.request_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }

    base_url = settings.resolved_llm_base_url
    if base_url is not None:
        model_kwargs["base_url"] = base_url

    return ChatOpenAI(**model_kwargs)


def convert_to_langchain_message(message: ChatMessage) -> BaseMessage:
    if message.role == ChatMessageRole.SYSTEM:
        return SystemMessage(content=message.content)
    if message.role == ChatMessageRole.USER:
        return HumanMessage(content=message.content)
    if message.role == ChatMessageRole.ASSISTANT:
        return AIMessage(content=message.content)

    raise AppException(
        code="LANGCHAIN_MESSAGE_ROLE_UNSUPPORTED",
        message="不支持的 LangChain 消息角色。",
        status_code=500,
    )


def build_langchain_chat_messages(
    user_message: str,
    *,
    history: Sequence[ChatMessage] | None = None,
) -> list[BaseMessage]:
    return [
        convert_to_langchain_message(message)
        for message in build_chat_messages(user_message, history=history)
    ]


def extract_langchain_reply(ai_message: Any) -> str:
    text = getattr(ai_message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(ai_message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    raise AppException(
        code="LLM_EMPTY_RESPONSE",
        message="模型返回了空内容",
        status_code=502,
    )


class LangChainChatModelService:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self._model = model

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            self._model = create_langchain_chat_model(self.settings)
        except ValueError as exc:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            ) from exc
        return self._model

    def generate_reply(
        self,
        user_message: str,
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> str:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_langchain_chat_messages(user_message, history=history)
        start_time = perf_counter()
        try:
            response = self._get_model().invoke(messages)
            reply = extract_langchain_reply(response)
        except AppException as exc:
            self._log_failure(exc, (perf_counter() - start_time) * 1000)
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(
                app_exception,
                (perf_counter() - start_time) * 1000,
                exc_info=True,
            )
            raise app_exception from exc

        self._log_success((perf_counter() - start_time) * 1000)
        return reply

    def _log_success(self, elapsed_ms: float) -> None:
        logger.info(
            "langchain_chat_model_succeeded provider=%s model=%s elapsed_ms=%.2f",
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
        )

    def _log_failure(
        self,
        app_exception: AppException,
        elapsed_ms: float,
        *,
        exc_info: bool = False,
    ) -> None:
        logger.warning(
            (
                "langchain_chat_model_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )


def create_langchain_chat_model_service(settings: Settings) -> LangChainChatModelService:
    return LangChainChatModelService(settings)
