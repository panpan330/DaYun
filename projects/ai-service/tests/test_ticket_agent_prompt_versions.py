from app.agents.ticket_agent import (
    TICKET_AGENT_PROMPTS,
    TICKET_FIELD_EXTRACTION_PROMPT,
    TICKET_INTENT_CLASSIFICATION_PROMPT,
    TicketAgentPromptSpec,
    build_ticket_field_extraction_messages,
    build_ticket_intent_classification_messages,
    create_ticket_agent_model_dependencies,
    get_ticket_agent_prompt_spec,
)
from app.core.config import Settings
from tests.fakes import FakeChatCompletions, FakeOpenAICompatibleClient


def test_ticket_agent_prompt_specs_are_registered_by_name() -> None:
    assert TICKET_AGENT_PROMPTS == {
        "ticket_intent_classification": TICKET_INTENT_CLASSIFICATION_PROMPT,
        "ticket_field_extraction": TICKET_FIELD_EXTRACTION_PROMPT,
    }
    assert (
        get_ticket_agent_prompt_spec("ticket_intent_classification")
        is TICKET_INTENT_CLASSIFICATION_PROMPT
    )
    assert (
        get_ticket_agent_prompt_spec("ticket_field_extraction")
        is TICKET_FIELD_EXTRACTION_PROMPT
    )


def test_ticket_agent_prompt_specs_use_stable_v1_versions() -> None:
    assert TICKET_INTENT_CLASSIFICATION_PROMPT.name == "ticket_intent_classification"
    assert (
        TICKET_INTENT_CLASSIFICATION_PROMPT.version
        == "ticket_intent_classification:v1"
    )
    assert TICKET_INTENT_CLASSIFICATION_PROMPT.system_prompt.strip()
    assert TICKET_INTENT_CLASSIFICATION_PROMPT.description

    assert TICKET_FIELD_EXTRACTION_PROMPT.name == "ticket_field_extraction"
    assert TICKET_FIELD_EXTRACTION_PROMPT.version == "ticket_field_extraction:v1"
    assert TICKET_FIELD_EXTRACTION_PROMPT.system_prompt.strip()
    assert TICKET_FIELD_EXTRACTION_PROMPT.description


def test_intent_message_builder_can_use_explicit_prompt_spec() -> None:
    prompt_spec = TicketAgentPromptSpec(
        name="ticket_intent_classification",
        version="ticket_intent_classification:v2",
        system_prompt="custom intent system prompt",
        description="test prompt override",
    )

    messages = build_ticket_intent_classification_messages(
        "hello",
        prompt_spec=prompt_spec,
    )

    assert messages[0] == {
        "role": "system",
        "content": "custom intent system prompt",
    }
    assert "JSON Schema:" in messages[1]["content"]
    assert "hello" in messages[1]["content"]


def test_field_message_builder_can_use_explicit_prompt_spec() -> None:
    prompt_spec = TicketAgentPromptSpec(
        name="ticket_field_extraction",
        version="ticket_field_extraction:v2",
        system_prompt="custom field system prompt",
        description="test prompt override",
    )

    messages = build_ticket_field_extraction_messages(
        {
            "normalized_message": "hello",
            "intent": "ticket_request",
            "ticket_need_source": "explicit_user_request",
        },
        prompt_spec=prompt_spec,
    )

    assert messages[0] == {
        "role": "system",
        "content": "custom field system prompt",
    }
    assert "JSON Schema:" in messages[1]["content"]
    assert "ticket_need_source" in messages[1]["content"]
    assert "hello" in messages[1]["content"]


def test_real_llm_dependency_factory_can_propagate_prompt_specs() -> None:
    intent_prompt_spec = TicketAgentPromptSpec(
        name="ticket_intent_classification",
        version="ticket_intent_classification:v2",
        system_prompt="custom intent system prompt",
        description="test intent prompt",
    )
    field_prompt_spec = TicketAgentPromptSpec(
        name="ticket_field_extraction",
        version="ticket_field_extraction:v2",
        system_prompt="custom field system prompt",
        description="test field prompt",
    )

    dependencies = create_ticket_agent_model_dependencies(
        "real_llm",
        settings=Settings(llm_api_key="test-key", _env_file=None),
        client=FakeOpenAICompatibleClient(FakeChatCompletions()),
        intent_prompt_spec=intent_prompt_spec,
        field_prompt_spec=field_prompt_spec,
    )

    assert dependencies["intent_classifier"].prompt_spec is intent_prompt_spec
    assert dependencies["field_extractor"].prompt_spec is field_prompt_spec
