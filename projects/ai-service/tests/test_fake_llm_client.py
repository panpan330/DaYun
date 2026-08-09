import pytest
from openai import BadRequestError

from tests.fakes import (
    FakeChatCompletions,
    FakeOpenAICompatibleClient,
    make_status_error,
    make_stream_chunk,
    make_usage,
)


def test_fake_chat_completions_returns_configured_content() -> None:
    completions = FakeChatCompletions(
        content="模型回复",
        usage=make_usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    client = FakeOpenAICompatibleClient(completions)

    completion = client.chat.completions.create(
        model="qwen-test",
        messages=[{"role": "user", "content": "你好"}],
    )

    assert completion.choices[0].message.content == "模型回复"
    assert completion.usage.prompt_tokens == 3
    assert completions.last_call["model"] == "qwen-test"
    assert completions.last_call["messages"] == [{"role": "user", "content": "你好"}]


def test_fake_chat_completions_records_stream_calls() -> None:
    completions = FakeChatCompletions(
        stream_chunks=[
            make_stream_chunk("你"),
            make_stream_chunk("好"),
        ]
    )
    client = FakeOpenAICompatibleClient(completions)

    stream = client.chat.completions.create(
        model="qwen-test",
        messages=[{"role": "user", "content": "你好"}],
        stream=True,
    )

    assert [chunk.choices[0].delta.content for chunk in stream] == ["你", "好"]
    assert completions.last_call["stream"] is True


def test_fake_chat_completions_can_raise_configured_error() -> None:
    completions = FakeChatCompletions(error=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        completions.create(model="qwen-test", messages=[])

    assert completions.last_call["model"] == "qwen-test"


def test_fake_chat_completions_last_call_requires_a_call() -> None:
    completions = FakeChatCompletions()

    with pytest.raises(AssertionError, match="was not called"):
        _ = completions.last_call


def test_make_status_error_builds_openai_status_error() -> None:
    error = make_status_error(BadRequestError, 400, message="bad request")

    assert isinstance(error, BadRequestError)
    assert error.status_code == 400
