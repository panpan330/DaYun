import json
import logging
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import AppException
from app.schemas.structured import TicketExtraction, get_ticket_extraction_json_schema
from app.services.llm_client import create_openai_compatible_client
from app.services.llm_service import (
    extract_first_reply,
    extract_token_usage,
    map_openai_error_to_app_exception,
)


logger = logging.getLogger(__name__)

TICKET_EXTRACTION_SYSTEM_PROMPT = (
    "你是客服工单字段抽取器。"
    "你必须只返回合法 JSON，不要返回 Markdown，不要返回解释文字。"
    "JSON 字段必须包含 intent、order_id、summary、urgency、need_human_review。"
    "intent 只能是 refund、order_query、logistics、complaint、unknown。"
    "urgency 只能是 low、normal、high。"
    "如果用户没有提到订单号，order_id 必须是 null。"
    "如果内容涉及强烈投诉、威胁升级、金额纠纷或你不确定，need_human_review 必须是 true。"
)


def build_ticket_extraction_messages(user_message: str) -> list[dict[str, str]]:
    schema_text = json.dumps(
        get_ticket_extraction_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {
            "role": "system",
            "content": TICKET_EXTRACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "请把下面的用户消息抽取成 JSON。\n"
                f"JSON Schema:\n{schema_text}\n\n"
                f"用户消息:\n{user_message}"
            ),
        },
    ]


def parse_ticket_extraction_json(raw_json: str) -> TicketExtraction:
    if not isinstance(raw_json, str) or not raw_json.strip():
        raise AppException(
            code="STRUCTURED_OUTPUT_EMPTY",
            message="模型没有返回可解析的结构化内容",
            status_code=502,
        )

    try:
        return TicketExtraction.model_validate_json(raw_json)
    except ValidationError as exc:
        raise AppException(
            code="STRUCTURED_OUTPUT_VALIDATION_FAILED",
            message="模型结构化输出校验失败，请稍后重试。",
            status_code=502,
            details=exc.errors(include_url=False),
        ) from exc


class StructuredOutputService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            self._client = create_openai_compatible_client(self.settings)
        except ValueError as exc:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            ) from exc
        return self._client

    def _log_success(
        self,
        elapsed_ms: float,
        completion: Any,
        result: TicketExtraction,
    ) -> None:
        usage = extract_token_usage(completion)
        logger.info(
            (
                "structured_ticket_extraction_succeeded provider=%s model=%s "
                "elapsed_ms=%.2f intent=%s urgency=%s need_human_review=%s "
                "prompt_tokens=%s completion_tokens=%s total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
            result.intent.value,
            result.urgency.value,
            result.need_human_review,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
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
                "structured_ticket_extraction_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
            exc_info=exc_info,
        )

    def extract_ticket(self, user_message: str) -> TicketExtraction:
        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_ticket_extraction_messages(user_message)
        start_time = perf_counter()
        try:
            completion = self._get_client().chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_reply = extract_first_reply(completion)
            result = parse_ticket_extraction_json(raw_reply)
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

        self._log_success((perf_counter() - start_time) * 1000, completion, result)
        return result


def create_structured_output_service(settings: Settings) -> StructuredOutputService:
    return StructuredOutputService(settings)
