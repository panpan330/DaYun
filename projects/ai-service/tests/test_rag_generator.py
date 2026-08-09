import logging

import pytest

from app.core.config import Settings
from app.core.exceptions import AppException
from app.rag.documents import RetrievedChunk
from app.rag.generator import (
    RAG_NO_CONTEXT_REPLY,
    RAG_NO_CONTEXT_SUGGESTIONS,
    RagAnswerStatus,
    RagAnswerService,
    RagNoContextReason,
    build_grounded_rag_answer,
    build_no_context_rag_answer,
    build_rag_citation,
    build_rag_citations,
    build_rag_context,
    build_rag_messages,
    build_rag_user_prompt,
    format_retrieved_chunk_for_context,
)
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient, make_usage


def make_retrieved_chunk(**overrides) -> RetrievedChunk:
    payload = {
        "point_id": "point-1",
        "chunk_id": "order_shipping_policy_chunk_0001",
        "content": "订单付款后通常会在 24 小时内发货。",
        "metadata": {
            "source": "order-shipping-policy.md",
            "title": "订单发货规则",
            "section": "正常发货时效",
        },
        "score": 0.91,
    }
    payload.update(overrides)
    return RetrievedChunk(**payload)


def make_service(completions: FakeChatCompletions) -> RagAnswerService:
    return RagAnswerService(
        Settings(
            llm_api_key="test-key",
            llm_provider="test-provider",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeOpenAICompatibleClient(completions),
    )


def test_format_retrieved_chunk_for_context_includes_metadata_and_content() -> None:
    chunk = make_retrieved_chunk()

    text = format_retrieved_chunk_for_context(1, chunk)

    assert "[资料 1]" in text
    assert "source: order-shipping-policy.md" in text
    assert "title: 订单发货规则" in text
    assert "section: 正常发货时效" in text
    assert "chunk_id: order_shipping_policy_chunk_0001" in text
    assert "score: 0.9100" in text
    assert "订单付款后通常会在 24 小时内发货。" in text


def test_format_retrieved_chunk_for_context_rejects_invalid_index() -> None:
    with pytest.raises(ValueError, match="index"):
        format_retrieved_chunk_for_context(0, make_retrieved_chunk())


def test_build_rag_context_numbers_chunks() -> None:
    chunks = [
        make_retrieved_chunk(chunk_id="chunk-1", content="第一段资料。", score=0.9),
        make_retrieved_chunk(chunk_id="chunk-2", content="第二段资料。", score=0.8),
    ]

    context = build_rag_context(chunks)

    assert "[资料 1]" in context
    assert "[资料 2]" in context
    assert "第一段资料。" in context
    assert "第二段资料。" in context


def test_build_rag_context_returns_empty_string_without_chunks() -> None:
    assert build_rag_context([]) == ""


def test_build_rag_user_prompt_contains_question_context_and_rules() -> None:
    prompt = build_rag_user_prompt(
        "订单多久发货？",
        [make_retrieved_chunk()],
    )

    assert "只能使用检索资料中的信息回答" in prompt
    assert "如果检索资料不足以回答" in prompt
    assert "订单多久发货？" in prompt
    assert "订单付款后通常会在 24 小时内发货。" in prompt
    assert "不要编造文件名、链接或不存在的出处" in prompt
    assert "最终引用来源由后端根据检索资料单独返回" in prompt


def test_build_rag_user_prompt_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        build_rag_user_prompt("   ", [make_retrieved_chunk()])


def test_build_rag_messages_uses_system_and_user_messages() -> None:
    messages = build_rag_messages("订单多久发货？", [make_retrieved_chunk()])

    assert messages[0]["role"] == "system"
    assert "只能根据后端提供的检索资料回答" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "检索资料" in messages[1]["content"]


def test_build_rag_citation_uses_backend_retrieved_metadata() -> None:
    chunk = make_retrieved_chunk()

    citation = build_rag_citation(1, chunk)

    assert citation.source_index == 1
    assert citation.source == "order-shipping-policy.md"
    assert citation.title == "订单发货规则"
    assert citation.section == "正常发货时效"
    assert citation.chunk_id == "order_shipping_policy_chunk_0001"
    assert citation.score == 0.91


def test_build_rag_citation_uses_fallback_for_missing_source() -> None:
    chunk = make_retrieved_chunk(metadata={"title": "", "section": ""})

    citation = build_rag_citation(1, chunk)

    assert citation.source == "unknown-source"
    assert citation.title is None
    assert citation.section is None


def test_build_rag_citation_rejects_invalid_index() -> None:
    with pytest.raises(ValueError, match="index"):
        build_rag_citation(0, make_retrieved_chunk())


def test_build_rag_citations_numbers_retrieved_chunks() -> None:
    citations = build_rag_citations(
        [
            make_retrieved_chunk(chunk_id="chunk-1", score=0.9),
            make_retrieved_chunk(chunk_id="chunk-2", score=0.8),
        ]
    )

    assert [citation.source_index for citation in citations] == [1, 2]
    assert [citation.chunk_id for citation in citations] == ["chunk-1", "chunk-2"]


def test_build_no_context_rag_answer_returns_structured_fallback() -> None:
    result = build_no_context_rag_answer()

    assert result.answer == RAG_NO_CONTEXT_REPLY
    assert result.status == RagAnswerStatus.NO_CONTEXT
    assert result.no_context_reason == RagNoContextReason.NO_RETRIEVED_CHUNKS
    assert result.citations == []
    assert result.suggestions == list(RAG_NO_CONTEXT_SUGGESTIONS)


def test_build_grounded_rag_answer_returns_answered_status_and_citations() -> None:
    result = build_grounded_rag_answer(
        "订单通常会在付款后 24 小时内发货。",
        [make_retrieved_chunk()],
    )

    assert result.answer == "订单通常会在付款后 24 小时内发货。"
    assert result.status == RagAnswerStatus.ANSWERED
    assert result.no_context_reason is None
    assert result.suggestions == []
    assert len(result.citations) == 1
    assert result.citations[0].source == "order-shipping-policy.md"


def test_rag_answer_service_calls_model_with_retrieved_context() -> None:
    completions = FakeChatCompletions(content="  订单通常会在付款后 24 小时内发货。  ")
    service = make_service(completions)

    reply = service.generate_answer(
        "订单多久发货？",
        chunks=[make_retrieved_chunk()],
    )

    assert reply == "订单通常会在付款后 24 小时内发货。"
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "qwen-test"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert "订单多久发货？" in call["messages"][1]["content"]
    assert "订单付款后通常会在 24 小时内发货。" in call["messages"][1]["content"]


def test_rag_answer_service_returns_answer_with_backend_citations() -> None:
    completions = FakeChatCompletions(content="订单通常会在付款后 24 小时内发货。")
    service = make_service(completions)

    result = service.generate_answer_with_citations(
        "订单多久发货？",
        chunks=[make_retrieved_chunk()],
    )

    assert result.answer == "订单通常会在付款后 24 小时内发货。"
    assert result.status == RagAnswerStatus.ANSWERED
    assert result.no_context_reason is None
    assert result.suggestions == []
    assert len(result.citations) == 1
    assert result.citations[0].source_index == 1
    assert result.citations[0].source == "order-shipping-policy.md"
    assert result.citations[0].chunk_id == "order_shipping_policy_chunk_0001"
    assert len(completions.calls) == 1


def test_rag_answer_service_returns_empty_citations_without_context() -> None:
    completions = FakeChatCompletions(content="不应该被调用")
    service = make_service(completions)

    result = service.generate_answer_with_citations("公司年会规则是什么？", chunks=[])

    assert result.answer == RAG_NO_CONTEXT_REPLY
    assert result.status == RagAnswerStatus.NO_CONTEXT
    assert result.no_context_reason == RagNoContextReason.NO_RETRIEVED_CHUNKS
    assert result.citations == []
    assert result.suggestions == list(RAG_NO_CONTEXT_SUGGESTIONS)
    assert completions.calls == []


def test_rag_answer_service_returns_no_context_reply_without_calling_model() -> None:
    completions = FakeChatCompletions(content="不应该被调用")
    service = make_service(completions)

    reply = service.generate_answer("公司年会抽奖规则是什么？", chunks=[])

    assert reply == RAG_NO_CONTEXT_REPLY
    assert completions.calls == []


def test_rag_answer_service_rejects_blank_query() -> None:
    service = make_service(FakeChatCompletions())

    with pytest.raises(ValueError, match="query"):
        service.generate_answer("   ", chunks=[make_retrieved_chunk()])


def test_rag_answer_service_requires_api_key_when_context_exists() -> None:
    service = RagAnswerService(Settings(_env_file=None))

    with pytest.raises(AppException) as exc_info:
        service.generate_answer("订单多久发货？", chunks=[make_retrieved_chunk()])

    assert exc_info.value.code == "LLM_API_KEY_MISSING"


def test_rag_answer_service_maps_model_errors() -> None:
    service = make_service(FakeChatCompletions(error=RuntimeError("provider failed")))

    with pytest.raises(AppException) as exc_info:
        service.generate_answer("订单多久发货？", chunks=[make_retrieved_chunk()])

    assert exc_info.value.code == "LLM_CALL_FAILED"


def test_rag_answer_service_logs_success_without_user_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="app.rag.generator")
    completions = FakeChatCompletions(
        content="订单通常会在付款后 24 小时内发货。",
        usage=make_usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
    )
    service = make_service(completions)

    service.generate_answer("订单多久发货？", chunks=[make_retrieved_chunk()])

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "rag_answer_succeeded provider=test-provider model=qwen-test" in message
        and "chunk_count=1" in message
        and "prompt_tokens=20" in message
        for message in messages
    )
    assert all("订单多久发货" not in message for message in messages)
