from datetime import datetime, timezone

import pytest

from app.core.config import Settings
from app.schemas.console_agent import ConsoleAgentConversation, ConsoleAgentConversationMessage
from app.services.console_agent_conversation_store import ConsoleAgentConversationStore
from app.services.console_agent_service import ConsoleAgentActor, ConsoleAgentService
from app.services.conversation_history import build_history_window


class RecordingClassifier:
    def __init__(self) -> None:
        self.seen_history: list[str] | None = None
        self.seen_messages: list[str] = []
        self.calls = 0

    def classify_intent(self, message: str, *, history: list[str] | None = None) -> dict:
        self.calls += 1
        self.seen_history = history
        self.seen_messages.append(message)
        return {"intent": "order_query", "reason": "recorded"}


class FakeGraph:
    def __init__(self, classifier: RecordingClassifier) -> None:
        self.classifier = classifier

    def invoke(self, *args, **kwargs):
        return {}


class FakeConversationStore:
    def __init__(self, conversation: ConsoleAgentConversation | None = None) -> None:
        self.conversation = conversation

    def get(self, *, actor, conversation_id: str) -> ConsoleAgentConversation | None:
        return self.conversation

    def append_exchange(self, **kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


def make_rounds(count: int) -> ConsoleAgentConversation:
    now = datetime.now(timezone.utc)
    messages = []
    for i in range(count):
        messages.append(ConsoleAgentConversationMessage(id=f"u{i}", role="user", content=f"历史用户{i}", created_at=now))
        messages.append(
            ConsoleAgentConversationMessage(id=f"a{i}", role="assistant", content=f"历史回复{i}", created_at=now)
        )
    return ConsoleAgentConversation(conversation_id="conv-h", title="历史", updated_at=now, messages=messages)


def _service(
    *,
    store: ConsoleAgentConversationStore | FakeConversationStore,
    classifier: RecordingClassifier,
) -> ConsoleAgentService:
    settings = Settings(
        _env_file=None,
        agent_redis_url="redis://redis.example:6379/0",
        agent_checkpoint_ttl_minutes=45,
        agent_checkpoint_key_prefix="test-agent",
        agent_history_max_rounds=10,
        agent_history_max_tokens=10000,
    )
    graph = FakeGraph(classifier)
    return ConsoleAgentService(settings, graph=graph, conversation_store=store)  # type: ignore[arg-type]


def test_reply_feeds_recent_history_into_intent_classifier() -> None:
    classifier = RecordingClassifier()
    store = FakeConversationStore(make_rounds(12))
    service = _service(store=store, classifier=classifier)
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    # 直接调用内部窗口组装 + 模拟 reply 的注入路径
    window = service._build_history_window(actor=actor, conversation_id="conv-h")
    from app.agents.ticket_agent import build_ticket_intent_classification_messages

    messages = build_ticket_intent_classification_messages("当前问题", history=window)

    assert window[0] == "历史用户2"  # 最旧 2 轮被窗口截掉（12 轮 → 10 轮）
    assert "历史用户11" in window
    assert "最近对话：" in messages[1]["content"]
    assert "历史用户2" in messages[1]["content"]
    assert "当前问题" in messages[1]["content"]


def test_history_window_respects_round_and_token_budgets() -> None:
    rounds = [{"role": m.role, "content": m.content} for m in make_rounds(12).messages]

    window = build_history_window(rounds, max_rounds=10, max_tokens=10000)

    assert len(window) == 19  # 10 轮 20 条，末尾 assistant 去掉
    assert window[0] == "历史用户2"
    assert window[-1] == "历史用户11"


def test_no_history_when_conversation_missing() -> None:
    service = _service(store=FakeConversationStore(None), classifier=RecordingClassifier())
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("customer",))

    window = service._build_history_window(actor=actor, conversation_id="conv-missing")

    assert window == []


def test_history_not_fed_to_rule_based_classifier() -> None:
    from app.agents.ticket_agent import RuleBasedTicketIntentClassifier

    classifier = RuleBasedTicketIntentClassifier()
    result = classifier.classify_intent("查订单", history=["历史用户0", "历史回复0"])

    assert result["intent"] == "order_query"


class RecordingGraph:
    def __init__(self) -> None:
        self.last_input: dict | None = None

    def invoke(self, initial_state: dict, **kwargs) -> dict:
        self.last_input = initial_state
        return {}

    def get_state(self, config) -> object:
        return type("Snapshot", (), {"values": {}, "next": ()})()


class RecordingStreamGraph(RecordingGraph):
    def stream(self, initial_state: dict, **kwargs):
        self.last_input = initial_state
        return iter(())


def test_reply_passes_actor_roles_into_thread() -> None:
    from app.core.config import Settings
    from app.services.console_agent_service import ConsoleAgentActor, ConsoleAgentService

    graph = RecordingGraph()
    settings = Settings(
        _env_file=None,
        agent_redis_url="redis://redis.example:6379/0",
        agent_checkpoint_ttl_minutes=45,
        agent_checkpoint_key_prefix="test-agent",
    )
    service = ConsoleAgentService(
        settings,
        graph=graph,  # type: ignore[arg-type]
        conversation_store=FakeConversationStore(None),
    )
    actor = ConsoleAgentActor(user_id="U1001", tenant_id="default", roles=("staff",))

    service.reply(actor=actor, conversation_id="conv-1", message="查订单")

    assert graph.last_input is not None
    assert graph.last_input.get("actor_roles") == ("staff",)
    assert graph.last_input.get("actor_tenant_id") == "default"
