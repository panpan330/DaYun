from app.core.config import Settings
from app.core.config_safety import (
    build_safe_settings_snapshot,
    build_secret_configuration_checks,
    find_raw_secret_setting_names,
)


def test_safe_settings_snapshot_exposes_status_not_secret_values() -> None:
    settings = Settings(
        llm_api_key="sk-test-llm-secret",
        embedding_api_key="sk-test-embedding-secret",
        rerank_api_key="sk-test-rerank-secret",
        qdrant_api_key="qdrant-secret",
        milvus_token="milvus-secret",
        openai_api_key="sk-test-openai-secret",
        llm_base_url="https://llm.example.com/compatible-mode/v1",
        embedding_base_url="https://embedding.example.com/compatible-mode/v1",
        rerank_base_url="https://rerank.example.com",
        llm_input_cost_per_million_tokens=2.0,
        llm_output_cost_per_million_tokens=6.0,
        _env_file=None,
    )

    snapshot = build_safe_settings_snapshot(settings)
    serialized = str(snapshot)

    assert snapshot["llm.api_key_configured"] is True
    assert snapshot["llm.default_route_tier"] == "balanced"
    assert snapshot["llm.route_long_input_chars"] == 1200
    assert snapshot["llm.route_fast_keyword_count"] >= 1
    assert snapshot["llm.route_strong_keyword_count"] >= 1
    assert snapshot["llm.fallback_enabled"] is True
    assert snapshot["llm.fallback_model_configured"] is False
    assert snapshot["llm.fallback_tier"] == "balanced"
    assert snapshot["llm.fallback_error_code_count"] >= 1
    assert snapshot["llm.total_timeout_seconds"] == 45.0
    assert snapshot["llm.cost_control_enabled"] is True
    assert snapshot["llm.max_input_tokens_per_request"] == 6000
    assert snapshot["llm.max_total_tokens_per_request"] == 8000
    assert snapshot["llm.min_output_tokens"] == 128
    assert snapshot["llm.max_estimated_cost_configured"] is False
    assert snapshot["llm.disable_fallback_above_total_tokens_configured"] is True
    assert snapshot["rate_limit.enabled"] is True
    assert snapshot["rate_limit.window_seconds"] == 60
    assert snapshot["rate_limit.client_requests_per_window"] == 120
    assert snapshot["rate_limit.route_requests_per_window"] == 120
    assert snapshot["rate_limit.ai_requests_per_window"] == 60
    assert snapshot["rate_limit.tool_requests_per_window"] == 30
    assert snapshot["rate_limit.excluded_path_count"] == 2
    assert snapshot["sse.heartbeat_every_chunks"] == 2
    assert snapshot["embedding.api_key_configured"] is True
    assert snapshot["rerank.api_key_configured"] is True
    assert snapshot["qdrant.api_key_configured"] is True
    assert snapshot["milvus.token_configured"] is True
    assert snapshot["llm.pricing_configured"] is True
    assert snapshot["llm.pricing_currency"] == "USD"
    assert "sk-test-llm-secret" not in serialized
    assert "sk-test-embedding-secret" not in serialized
    assert "sk-test-rerank-secret" not in serialized
    assert "qdrant-secret" not in serialized
    assert "milvus-secret" not in serialized
    assert find_raw_secret_setting_names(snapshot) == []


def test_secret_configuration_checks_require_llm_key_only_for_real_llm() -> None:
    settings = Settings(
        ticket_agent_model_mode="real_llm",
        llm_api_key=None,
        openai_api_key=None,
        _env_file=None,
    )

    checks = {check.name: check for check in build_secret_configuration_checks(settings)}

    assert checks["llm_api_key"].required is True
    assert checks["llm_api_key"].configured is False
    assert checks["llm_api_key"].readiness_status == "not_configured"
    assert checks["embedding_api_key"].required is False
    assert checks["embedding_api_key"].readiness_status == "skipped"
    assert checks["rerank_api_key"].required is False
    assert checks["rerank_api_key"].readiness_status == "skipped"


def test_secret_configuration_checks_accept_real_llm_with_fallback_openai_key() -> None:
    settings = Settings(
        ticket_agent_model_mode="real_llm",
        llm_api_key="   ",
        openai_api_key="legacy-openai-key",
        _env_file=None,
    )

    checks = {check.name: check for check in build_secret_configuration_checks(settings)}

    assert checks["llm_api_key"].required is True
    assert checks["llm_api_key"].configured is True
    assert checks["llm_api_key"].readiness_status == "configured"


def test_secret_configuration_checks_skip_llm_key_for_rule_based_mode() -> None:
    settings = Settings(
        ticket_agent_model_mode="rule_based",
        llm_api_key=None,
        openai_api_key=None,
        _env_file=None,
    )

    checks = {check.name: check for check in build_secret_configuration_checks(settings)}

    assert checks["llm_api_key"].required is False
    assert checks["llm_api_key"].readiness_status == "skipped"


def test_find_raw_secret_setting_names_reports_only_raw_secret_fields() -> None:
    forbidden = find_raw_secret_setting_names(
        {
            "llm_api_key": "sk-test-secret",
            "rerank-api-key": "rerank-secret",
            "milvus.token": "milvus-secret",
            "llm.api_key_configured": True,
            "app.name": "AI Service",
        }
    )

    assert forbidden == ["llm_api_key", "milvus_token", "rerank_api_key"]
