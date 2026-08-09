from openai import OpenAI

from app.core.config import Settings


def create_openai_compatible_client(settings: Settings) -> OpenAI:
    api_key = settings.resolved_llm_api_key
    if api_key is None:
        raise ValueError("LLM_API_KEY is not configured")

    client_kwargs: dict[str, object] = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": settings.request_timeout_seconds,
    }

    base_url = settings.resolved_llm_base_url
    if base_url is not None:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)
