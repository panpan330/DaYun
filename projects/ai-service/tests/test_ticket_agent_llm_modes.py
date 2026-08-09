import json

import pytest

from app.agents.ticket_agent import (
    FakeLLMTicketFieldExtractor,
    FakeLLMTicketIntentClassifier,
    LLMTicketFieldExtractor,
    LLMTicketIntentClassifier,
    build_ticket_agent_graph_for_model_mode,
    build_ticket_agent_input,
    create_ticket_agent_model_dependencies,
    extract_ticket_fields_node,
)
from app.core.config import Settings
from app.core.exceptions import AppException
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient


def test_create_ticket_agent_model_dependencies_defaults_to_rule_based() -> None:
    dependencies = create_ticket_agent_model_dependencies(
        settings=Settings(_env_file=None)
    )

    assert dependencies == {
        "mode": "rule_based",
        "intent_classifier": None,
        "field_extractor": None,
        "emotion_classifier": None,
    }


def test_create_ticket_agent_model_dependencies_can_use_fake_llm_mode_from_settings() -> None:
    dependencies = create_ticket_agent_model_dependencies(
        settings=Settings(ticket_agent_model_mode="fake_llm", _env_file=None)
    )

    assert dependencies["mode"] == "fake_llm"
    assert isinstance(dependencies["intent_classifier"], FakeLLMTicketIntentClassifier)
    assert isinstance(dependencies["field_extractor"], FakeLLMTicketFieldExtractor)


def test_create_ticket_agent_model_dependencies_real_mode_requires_api_key() -> None:
    with pytest.raises(AppException) as exc_info:
        create_ticket_agent_model_dependencies(
            "real_llm",
            settings=Settings(llm_api_key=None, openai_api_key=None, _env_file=None),
        )

    assert exc_info.value.code == "LLM_API_KEY_MISSING"


def test_create_ticket_agent_model_dependencies_real_mode_does_not_call_client_when_building() -> None:
    completions = FakeChatCompletions()
    dependencies = create_ticket_agent_model_dependencies(
        "real_llm",
        settings=Settings(llm_api_key="test-key", _env_file=None),
        client=FakeOpenAICompatibleClient(completions),
    )

    assert dependencies["mode"] == "real_llm"
    assert isinstance(dependencies["intent_classifier"], LLMTicketIntentClassifier)
    assert isinstance(dependencies["field_extractor"], LLMTicketFieldExtractor)
    assert completions.calls == []


def test_build_ticket_agent_graph_for_model_mode_keeps_rule_based_default_without_llm_call() -> None:
    completions = FakeChatCompletions()
    graph = build_ticket_agent_graph_for_model_mode(
        settings=Settings(_env_file=None),
        client=FakeOpenAICompatibleClient(completions),
    )

    result = graph.invoke(build_ticket_agent_input("hello"))

    assert result["intent"] == "smalltalk"
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_direct_answer",
    ]
    assert completions.calls == []


def test_build_ticket_agent_graph_for_model_mode_can_run_fake_llm_without_api_key() -> None:
    graph = build_ticket_agent_graph_for_model_mode(
        mode="fake_llm",
        settings=Settings(llm_api_key=None, openai_api_key=None, _env_file=None),
    )

    result = graph.invoke(build_ticket_agent_input("hello"))

    assert result["intent"] == "smalltalk"
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_direct_answer",
    ]


def test_fake_llm_field_extractor_uses_same_validation_path_as_llm_fields() -> None:
    extractor = FakeLLMTicketFieldExtractor()

    update = extract_ticket_fields_node(
        {
            "normalized_message": "会员等级政策没有查到，请帮我转人工解释",
            "ticket_need_source": "rag_no_context",
            "rag_answer_status": "no_context",
        },
        extractor=extractor,
    )

    assert update["ticket_field_extraction_source"] == "fake_llm"
    assert update["ticket_fields"]["issue_type"] == "policy_gap"
    assert update["ticket_fields_complete"] is True
    assert update["missing_ticket_fields"] == []


def test_build_ticket_agent_graph_for_model_mode_real_llm_can_be_tested_with_fake_client() -> None:
    completions = FakeChatCompletions(
        content=json.dumps(
            {
                "intent": "smalltalk",
                "reason": "用户只是问候，不需要进入业务流程。",
            },
            ensure_ascii=False,
        )
    )
    graph = build_ticket_agent_graph_for_model_mode(
        mode="real_llm",
        settings=Settings(
            llm_api_key="test-key",
            llm_model="qwen-test",
            _env_file=None,
        ),
        client=FakeOpenAICompatibleClient(completions),
    )

    result = graph.invoke(build_ticket_agent_input("hello"))

    assert result["intent"] == "smalltalk"
    assert result["intent_reason"] == "用户只是问候，不需要进入业务流程。"
    assert result["node_history"] == [
        "normalize_user_input",
        "emotion_recognition",
        "classify_intent",
        "build_direct_answer",
    ]
    assert len(completions.calls) == 1
    assert completions.last_call["model"] == "qwen-test"
    assert completions.last_call["response_format"] == {"type": "json_object"}
