"""Telemetry tests for the RAG answer model call.

Verifies that ``RagAnswerService.generate_answer`` wraps its LLM call in an
``llm.call`` span (per spec 3.1: llm.call covers classification / field
extraction / answer call sites) with model/provider/prompt_name and token
attributes. Uses an in-memory exporter, no real Collector / OTLP network.
"""

from typing import Any, Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.config import Settings
from app.core.telemetry import shutdown_telemetry
from app.core.trace import reset_trace_id, set_trace_id
from app.rag.documents import RetrievedChunk
from app.rag.generator import RagAnswerService
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient, make_usage


def make_retrieved_chunk(**overrides: Any) -> RetrievedChunk:
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


def _reset_tracer_provider() -> None:
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    once = getattr(trace, "_TRACER_PROVIDER_SET_ONCE", None)
    if once is not None:
        once._done = False  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_tracer_provider() -> None:
    _reset_tracer_provider()
    yield
    _reset_tracer_provider()


@pytest.fixture()
def otel_exporter() -> Generator[InMemorySpanExporter, None, None]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    token = set_trace_id("test-trace-123")
    yield exporter
    reset_trace_id(token)
    shutdown_telemetry()


def test_generate_answer_emits_llm_call_span_with_model_attributes(
    otel_exporter: InMemorySpanExporter,
) -> None:
    completions = FakeChatCompletions(
        content="订单通常会在付款后 24 小时内发货。",
        usage=make_usage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
    )
    service = make_service(completions)

    reply = service.generate_answer(
        "订单多久发货？",
        chunks=[make_retrieved_chunk()],
    )

    llm_spans = [s for s in otel_exporter.get_finished_spans() if s.name == "llm.call"]
    assert len(llm_spans) == 1
    attrs = dict(llm_spans[0].attributes)
    assert attrs["model"] == "qwen-test"
    assert attrs["provider"] == "test-provider"
    assert attrs["prompt_name"] == "rag_answer"
    assert attrs["prompt_tokens"] == 20
    assert attrs["completion_tokens"] == 8
    assert attrs["total_tokens"] == 28
    assert reply == "订单通常会在付款后 24 小时内发货。"
    assert len(completions.calls) == 1


def test_generate_answer_llm_span_when_telemetry_disabled() -> None:
    # 未设置 SDK provider 时辅助函数 no-op，不抛异常且不产生 span。
    completions = FakeChatCompletions(content="回复")
    service = make_service(completions)

    reply = service.generate_answer("订单多久发货？", chunks=[make_retrieved_chunk()])

    assert reply == "回复"


def test_generate_answer_emits_span_on_failure_path(
    otel_exporter: InMemorySpanExporter,
) -> None:
    from app.core.exceptions import AppException

    service = make_service(FakeChatCompletions(error=RuntimeError("provider failed")))

    with pytest.raises(AppException):
        service.generate_answer("订单多久发货？", chunks=[make_retrieved_chunk()])

    llm_spans = [s for s in otel_exporter.get_finished_spans() if s.name == "llm.call"]
    assert len(llm_spans) == 1
    attrs = dict(llm_spans[0].attributes)
    assert attrs["model"] == "qwen-test"
