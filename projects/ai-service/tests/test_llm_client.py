import pytest

from app.core.config import Settings
from app.services import llm_client


class FakeOpenAI:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_create_openai_compatible_client_uses_llm_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    settings = Settings(
        llm_api_key="llm-test-key",
        llm_base_url=" https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 ",
        llm_max_retries=3,
        request_timeout_seconds=12.5,
        _env_file=None,
    )

    client = llm_client.create_openai_compatible_client(settings)

    assert isinstance(client, FakeOpenAI)
    assert client.kwargs == {
        "api_key": "llm-test-key",
        "base_url": "https://example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "max_retries": 0,
        "timeout": 12.5,
    }


def test_create_openai_compatible_client_works_without_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)
    settings = Settings(llm_api_key="llm-test-key", _env_file=None)

    client = llm_client.create_openai_compatible_client(settings)

    assert isinstance(client, FakeOpenAI)
    assert client.kwargs == {
        "api_key": "llm-test-key",
        "max_retries": 0,
        "timeout": 30.0,
    }


def test_create_openai_compatible_client_requires_api_key() -> None:
    settings = Settings(llm_api_key="", openai_api_key="", _env_file=None)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        llm_client.create_openai_compatible_client(settings)
