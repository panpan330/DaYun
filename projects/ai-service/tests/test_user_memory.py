"""Tests for user memory store and rule-based fact extraction."""

from app.services.console_agent_service import ConsoleAgentActor
from app.services.user_memory import (
    MemoryService,
    UserMemoryFact,
    UserMemoryStore,
    extract_user_facts,
    extract_user_facts_llm,
    normalize_fact,
)

ACTOR = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))


class _FakeRedis:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def get(self, key: str):
        return self._d.get(key)

    def set(self, key: str, value: str, ex=None) -> None:
        self._d[key] = value

    def delete(self, key: str) -> None:
        self._d.pop(key, None)


def _make_store() -> UserMemoryStore:
    return UserMemoryStore(memory_service=MemoryService(client=_FakeRedis()))


def test_normalize_fact_ignores_whitespace_case():
    assert normalize_fact(" 我叫 小明 ") == normalize_fact("我叫小明")


def test_rule_extraction_names_address_child():
    facts = extract_user_facts("我叫小明，住在朝阳区，孩子上小学", "conv-1")
    texts = [f.fact for f in facts]
    assert any("小明" in t for t in texts)
    assert any("朝阳区" in t for t in texts)
    assert any("小学" in t for t in texts)


def test_rule_extraction_phone():
    facts = extract_user_facts("我的手机号是13800138000", "conv-2")
    assert any("13800138000" in f.fact for f in facts)


def test_rule_extraction_no_facts():
    assert extract_user_facts("你们发货要多久", "conv-3") == []


def test_store_roundtrip_and_clear():
    store = _make_store()
    store.store_facts(
        ACTOR,
        [UserMemoryFact(fact="用户称呼：小明", source_conversation_id="c1", extracted_at="2026-08-15T00:00:00", confidence=0.8)],
    )
    facts = store.get_facts(ACTOR)
    assert [f["fact"] for f in facts] == ["用户称呼：小明"]
    store.clear(ACTOR)
    assert store.get_facts(ACTOR) == []


def test_store_dedupes_latest():
    store = _make_store()
    store.store_facts(
        ACTOR,
        [UserMemoryFact(fact="用户称呼：小明", source_conversation_id="c1", extracted_at="2026-08-15T00:00:00", confidence=0.8)],
    )
    store.store_facts(
        ACTOR,
        [UserMemoryFact(fact="用户称呼：小刚", source_conversation_id="c2", extracted_at="2026-08-15T01:00:00", confidence=0.9)],
    )
    facts = store.get_facts(ACTOR)
    assert [f["fact"] for f in facts] == ["用户称呼：小刚"]


class _SettingsLike:
    def __init__(self, has_llm_api_key: bool) -> None:
        self.has_llm_api_key = has_llm_api_key


def test_llm_extraction_parses_structured_output():
    class _FakeClient:
        def generate_reply(self, user_message: str, **kwargs) -> str:
            return '{"facts": [{"text": "用户称呼：小红", "confidence": 0.95}]}'

    facts = extract_user_facts_llm(
        "我叫小红", "conv-9", _SettingsLike(has_llm_api_key=True), client=_FakeClient()
    )
    assert facts and facts[0].fact == "用户称呼：小红"


def test_llm_extraction_without_key_returns_empty():
    facts = extract_user_facts_llm(
        "我叫小红", "conv-9", _SettingsLike(has_llm_api_key=False), client=None
    )
    assert facts == []
