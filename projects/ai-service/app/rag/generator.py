import logging
from collections.abc import Sequence
from enum import Enum
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.exceptions import AppException
from app.rag.documents import RetrievedChunk
from app.services.llm_client import create_openai_compatible_client
from app.services.llm_service import (
    extract_first_reply,
    extract_token_usage,
    map_openai_error_to_app_exception,
)


logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = (
    "你是一个企业知识库 RAG 问答助手。"
    "你只能根据后端提供的检索资料回答用户问题。"
    "如果资料不足以回答，必须明确说明当前知识库资料不足，不能编造。"
    "回答要使用中文，表达清楚，适合客服或业务人员阅读。"
)

RAG_NO_CONTEXT_REPLY = "当前知识库没有找到足够相关的资料，无法根据知识库回答这个问题。"

RAG_NO_CONTEXT_SUGGESTIONS = (
    "换一种更具体的问法，例如补充订单、退款、物流或账号安全等关键词。",
    "确认问题是否属于当前知识库覆盖范围。",
    "如果这是新政策或新问题，可以记录为待补充知识。",
)


def format_retrieved_chunk_for_context(index: int, chunk: RetrievedChunk) -> str:
    if index <= 0:
        raise ValueError("context chunk index must be greater than 0")

    source = chunk.metadata.get("source", "unknown-source")
    section = chunk.metadata.get("section", "unknown-section")
    title = chunk.metadata.get("title", "unknown-title")

    return (
        f"[资料 {index}]\n"
        f"source: {source}\n"
        f"title: {title}\n"
        f"section: {section}\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"score: {chunk.score:.4f}\n"
        f"content:\n{chunk.content.strip()}"
    )


class RagCitation(BaseModel):
    source_index: int = Field(
        ge=1,
        description="One-based index matching the retrieved context block shown to the model.",
    )
    source: str = Field(
        min_length=1,
        description="Original source file or source identifier from chunk metadata.",
    )
    title: str | None = Field(
        default=None,
        description="Human-readable document title when available.",
    )
    section: str | None = Field(
        default=None,
        description="Section heading inside the source document when available.",
    )
    chunk_id: str = Field(
        min_length=1,
        description="Stable chunk id used to trace the answer back to stored knowledge.",
    )
    score: float = Field(
        description="Retriever similarity score for this chunk.",
    )


class RagAnswerStatus(str, Enum):
    ANSWERED = "answered"
    NO_CONTEXT = "no_context"
    PERMISSION_DENIED = "permission_denied"


class RagNoContextReason(str, Enum):
    NO_RETRIEVED_CHUNKS = "no_retrieved_chunks"


class RagAnswer(BaseModel):
    answer: str = Field(
        min_length=1,
        description="Grounded natural-language answer generated from retrieved context.",
    )
    status: RagAnswerStatus = Field(
        default=RagAnswerStatus.ANSWERED,
        description="Whether the service answered from retrieved context or returned a no-context fallback.",
    )
    citations: list[RagCitation] = Field(
        default_factory=list,
        description="Backend-generated citations for the retrieved chunks used as context.",
    )
    no_context_reason: RagNoContextReason | None = Field(
        default=None,
        description="Machine-readable reason when the service cannot answer from retrieved context.",
    )
    permission_denied: bool = Field(
        default=False,
        description="True when retrieval was blocked by access scope (permission_denied status).",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="User-facing next steps when the knowledge base has no usable context.",
    )


def _optional_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_metadata_text(value: Any, fallback: str) -> str:
    text = _optional_metadata_text(value)
    return text if text is not None else fallback


def build_rag_citation(index: int, chunk: RetrievedChunk) -> RagCitation:
    if index <= 0:
        raise ValueError("citation index must be greater than 0")

    return RagCitation(
        source_index=index,
        source=_required_metadata_text(
            chunk.metadata.get("source"),
            fallback="unknown-source",
        ),
        title=_optional_metadata_text(chunk.metadata.get("title")),
        section=_optional_metadata_text(chunk.metadata.get("section")),
        chunk_id=chunk.chunk_id,
        score=chunk.score,
    )


def build_rag_citations(chunks: Sequence[RetrievedChunk]) -> list[RagCitation]:
    return [
        build_rag_citation(index, chunk)
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_no_context_rag_answer() -> RagAnswer:
    return RagAnswer(
        answer=RAG_NO_CONTEXT_REPLY,
        status=RagAnswerStatus.NO_CONTEXT,
        citations=[],
        no_context_reason=RagNoContextReason.NO_RETRIEVED_CHUNKS,
        suggestions=list(RAG_NO_CONTEXT_SUGGESTIONS),
    )


def build_grounded_rag_answer(answer: str, chunks: Sequence[RetrievedChunk]) -> RagAnswer:
    return RagAnswer(
        answer=answer,
        status=RagAnswerStatus.ANSWERED,
        citations=build_rag_citations(chunks),
    )


def build_rag_context(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    return "\n\n".join(
        format_retrieved_chunk_for_context(index, chunk)
        for index, chunk in enumerate(chunks, start=1)
    )


def build_rag_user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")

    context = build_rag_context(chunks)
    if not context:
        context = "无可用检索资料。"

    return (
        "请根据下面的检索资料回答用户问题。\n\n"
        "回答规则：\n"
        "1. 只能使用检索资料中的信息回答。\n"
        "2. 如果检索资料不足以回答，直接说明资料不足，不要编造。\n"
        "3. 不要把资料编号、score 或 chunk_id 当成业务事实。\n"
        "4. 可以按需要在回答中提到资料编号，但不要编造文件名、链接或不存在的出处。\n"
        "5. 最终引用来源由后端根据检索资料单独返回。\n\n"
        f"用户问题：\n{normalized_query}\n\n"
        f"检索资料：\n{context}"
    )


def build_rag_messages(
    query: str,
    chunks: Sequence[RetrievedChunk],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_rag_user_prompt(query, chunks),
        },
    ]


class RagAnswerService:
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
        *,
        chunk_count: int,
        completion: Any,
    ) -> None:
        usage = extract_token_usage(completion)
        logger.info(
            (
                "rag_answer_succeeded provider=%s model=%s elapsed_ms=%.2f "
                "chunk_count=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s"
            ),
            self.settings.llm_provider,
            self.settings.llm_model,
            elapsed_ms,
            chunk_count,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    def _log_no_context(self) -> None:
        logger.info(
            "rag_answer_skipped reason=no_context provider=%s model=%s",
            self.settings.llm_provider,
            self.settings.llm_model,
        )

    def _log_failure(self, app_exception: AppException, elapsed_ms: float) -> None:
        logger.warning(
            (
                "rag_answer_failed code=%s provider=%s model=%s "
                "status_code=%s elapsed_ms=%.2f"
            ),
            app_exception.code,
            self.settings.llm_provider,
            self.settings.llm_model,
            app_exception.status_code,
            elapsed_ms,
        )

    def generate_answer(
        self,
        query: str,
        *,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")

        if not chunks:
            self._log_no_context()
            return RAG_NO_CONTEXT_REPLY

        if not self.settings.has_llm_api_key:
            raise AppException(
                code="LLM_API_KEY_MISSING",
                message="LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
                status_code=500,
            )

        messages = build_rag_messages(normalized_query, chunks)
        start_time = perf_counter()
        try:
            from app.agents.tracing_spans import set_span_attributes, start_llm_span

            with start_llm_span(
                model=self.settings.llm_model,
                provider=self.settings.llm_provider,
                prompt_name="rag_answer",
            ):
                completion = self._get_client().chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                )
                usage = extract_token_usage(completion)
                token_attributes: dict[str, int] = {}
                if usage.prompt_tokens is not None:
                    token_attributes["prompt_tokens"] = usage.prompt_tokens
                if usage.completion_tokens is not None:
                    token_attributes["completion_tokens"] = usage.completion_tokens
                if usage.total_tokens is not None:
                    token_attributes["total_tokens"] = usage.total_tokens
                if token_attributes:
                    set_span_attributes(token_attributes)
            reply = extract_first_reply(completion)
        except AppException as exc:
            self._log_failure(exc, (perf_counter() - start_time) * 1000)
            raise
        except Exception as exc:
            app_exception = map_openai_error_to_app_exception(exc)
            self._log_failure(app_exception, (perf_counter() - start_time) * 1000)
            raise app_exception from exc

        self._log_success(
            (perf_counter() - start_time) * 1000,
            chunk_count=len(chunks),
            completion=completion,
        )
        return reply

    def generate_answer_with_citations(
        self,
        query: str,
        *,
        chunks: Sequence[RetrievedChunk],
    ) -> RagAnswer:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be blank")

        if not chunks:
            self._log_no_context()
            return build_no_context_rag_answer()

        answer = self.generate_answer(normalized_query, chunks=chunks)
        return build_grounded_rag_answer(answer, chunks)


def create_rag_answer_service(settings: Settings) -> RagAnswerService:
    return RagAnswerService(settings)
