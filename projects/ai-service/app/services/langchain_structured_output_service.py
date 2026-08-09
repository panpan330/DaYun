import json
import logging
from time import perf_counter
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketExtraction, get_ticket_extraction_json_schema
from app.services.langchain_chat_model_service import create_langchain_chat_model
from app.services.llm_service import map_openai_error_to_app_exception
from app.services.structured_output_service import TICKET_EXTRACTION_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def build_langchain_ticket_extraction_messages(user_message: str) -> list[BaseMessage]:
    schema_text = json.dumps(
        get_ticket_extraction_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        SystemMessage(content=TICKET_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请把下面的用户消息抽取成结构化工单字段。\n"
                f"JSON Schema:\n{schema_text}\n\n"
                f"用户消息:\n{user_message}"
            )
        ),
    ]


def validate_langchain_ticket_extraction(raw_result: Any) -> TicketExtraction:
    if isinstance(raw_result, TicketExtraction):
        return raw_result

    try:
        return TicketExtraction.model_validate(raw_result)
    except ValidationError as exc:
        raise AppException(
            code="STRUCTURED_OUTPUT_VALIDATION_FAILED",
            message="模型结构化输出校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False),
        ) from exc


class LangChainStructuredOutputService:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self._model = model
        self._structured_model: Any | None = None

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

    def _get_structured_model(self) -> Any:
        if self._structured_model is None:
            self._structured_model = self._get_model().with_structured_output(
                TicketExtraction,
                method="json_mode",
            )
        return self._structured_model

    def extract_ticket(self, user_message: str) -> TicketExtraction:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_langchain_ticket_extraction_messages(user_message)
        start_time = perf_counter()
        try:
            raw_result = self._get_structured_model().invoke(messages)
            result = validate_langchain_ticket_extraction(raw_result)
        except AppException as exc:
            self._log_failure(exc, (perf_counter() - start_time) * 1000)
            raise
        except (ValidationError, OutputParserException) as exc:
            app_exception = AppException(
                code="STRUCTURED_OUTPUT_VALIDATION_FAILED",
                message="模型结构化输出校验失败，请稍后重试。",
                status_code=502,
            )
            self._log_failure(
                app_exception,
                (perf_counter() - start_time) * 1000,
                exc_info=True,
            )
            raise app_exception from exc
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(
                app_exception,
                (perf_counter() - start_time) * 1000,
                exc_info=True,
            )
            raise app_exception from exc

        self._log_success((perf_counter() - start_time) * 1000, result)
        return result

    def _log_success(self, elapsed_ms: float, result: TicketExtraction) -> None:
        logger.info(
            (
                "langchain_structured_ticket_extraction_succeeded "
                "provider=%s model=%s elapsed_ms=%.2f intent=%s urgency=%s "
                "need_human_review=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
            result.intent.value,
            result.urgency.value,
            result.need_human_review,
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
                "langchain_structured_ticket_extraction_failed code=%s "
                "provider=%s model=%s status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )


def create_langchain_structured_output_service(
    settings: Settings,
) -> LangChainStructuredOutputService:
    return LangChainStructuredOutputService(settings)
