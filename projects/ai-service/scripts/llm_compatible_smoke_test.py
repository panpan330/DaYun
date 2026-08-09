import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.llm_service import LLMChatService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or call an OpenAI-compatible LLM endpoint."
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="Actually call the configured model. This may incur cost.",
    )
    parser.add_argument(
        "--prompt",
        default="请用一句话解释 FastAPI 是什么。",
        help="Prompt to send when --call is used.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.has_llm_api_key:
        print("LLM_API_KEY is missing. Put your real key in local .env first.")
        return 2

    print(f"provider={settings.llm_provider}")
    print(f"model={settings.llm_model}")
    print(f"base_url_configured={settings.resolved_llm_base_url is not None}")
    print(f"max_retries={settings.llm_max_retries}")

    if not args.call:
        print("SDK configuration looks ready. No API call was made.")
        print("Run again with --call only when you intentionally want to call the model.")
        return 0

    print(LLMChatService(settings).generate_reply(args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
