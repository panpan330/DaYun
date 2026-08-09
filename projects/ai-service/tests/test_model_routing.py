from app.core.config import Settings
from app.core.model_routing import parse_route_keywords, route_llm_model


def test_route_llm_model_defaults_to_balanced_tier_and_legacy_model() -> None:
    settings = Settings(llm_model="qwen-default", _env_file=None)

    decision = route_llm_model(settings, operation="chat", input_text="解释 FastAPI")

    assert decision.provider == "openai-compatible"
    assert decision.model == "qwen-default"
    assert decision.tier == "balanced"
    assert decision.reason == "default_tier"
    assert decision.operation == "chat"


def test_route_llm_model_selects_fast_tier_for_simple_keyword_task() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_fast_model="qwen-fast",
        llm_route_fast_keywords="摘要,翻译",
        _env_file=None,
    )

    decision = route_llm_model(settings, operation="chat", input_text="帮我摘要这段文字")

    assert decision.model == "qwen-fast"
    assert decision.tier == "fast"
    assert decision.reason == "fast_keyword"


def test_route_llm_model_selects_strong_tier_for_complex_keyword_task() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_strong_model="qwen-strong",
        llm_route_strong_keywords="架构设计,生产事故",
        _env_file=None,
    )

    decision = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我做一下这个系统的架构设计",
    )

    assert decision.model == "qwen-strong"
    assert decision.tier == "strong"
    assert decision.reason == "strong_keyword"


def test_route_llm_model_selects_strong_tier_for_long_input() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_strong_model="qwen-strong",
        llm_route_long_input_chars=100,
        _env_file=None,
    )

    decision = route_llm_model(
        settings,
        operation="stream_chat",
        input_text="请分析：" + "业务流程" * 40,
    )

    assert decision.model == "qwen-strong"
    assert decision.tier == "strong"
    assert decision.reason == "long_input"


def test_route_llm_model_keeps_quality_operations_on_balanced_tier() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_fast_model="qwen-fast",
        llm_balanced_model="qwen-balanced",
        _env_file=None,
    )

    decision = route_llm_model(
        settings,
        operation="rag_answer",
        input_text="摘要订单退款知识库",
    )

    assert decision.model == "qwen-balanced"
    assert decision.tier == "balanced"
    assert decision.reason == "operation_requires_quality"


def test_route_llm_model_supports_explicit_preferred_tier() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_fast_model="qwen-fast",
        _env_file=None,
    )

    decision = route_llm_model(
        settings,
        operation="chat",
        input_text="帮我做复杂推理",
        preferred_tier="fast",
    )

    assert decision.model == "qwen-fast"
    assert decision.tier == "fast"
    assert decision.reason == "preferred_tier"


def test_route_llm_model_falls_back_to_legacy_model_when_tier_model_blank() -> None:
    settings = Settings(
        llm_model="qwen-default",
        llm_fast_model="   ",
        _env_file=None,
    )

    decision = route_llm_model(settings, operation="chat", input_text="摘要这段话")

    assert decision.model == "qwen-default"
    assert decision.tier == "fast"


def test_route_decision_log_fields_do_not_include_prompt_text() -> None:
    settings = Settings(llm_model="qwen-default", _env_file=None)

    decision = route_llm_model(settings, operation="chat", input_text="用户真实问题")
    fields = decision.to_log_fields()
    serialized = str(fields)

    assert fields["llm.route_tier"] == "balanced"
    assert fields["llm.route_operation"] == "chat"
    assert fields["llm.input_chars"] == len("用户真实问题")
    assert "用户真实问题" not in serialized


def test_parse_route_keywords_splits_common_separators_and_deduplicates() -> None:
    assert parse_route_keywords("摘要，翻译;摘要\n分类") == (
        "摘要",
        "翻译",
        "分类",
    )
