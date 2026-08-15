"""Cross-session user memory: rule extraction + Redis storage (facts only, never logs)."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import redis as redis_lib

logger = logging.getLogger(__name__)

_FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:我叫|我是|名字是)\s*([\u4e00-\u9fa5A-Za-z]{1,12})"), "用户称呼：{0}"),
    (re.compile(r"(?:住在|地址是|地址在)\s*([\u4e00-\u9fa5A-Za-z0-9]{2,30})"), "用户地址：{0}"),
    (re.compile(r"(?:孩子|小孩)(?:上|在读|今年读)\s*([\u4e00-\u9fa5A-Za-z0-9]{1,8})"), "用户孩子：{0}"),
    (re.compile(r"(?:手机号|电话|联系方式)\s*(?:是|为)?\s*([0-9]{11})"), "用户手机号：{0}"),
]


@dataclass
class UserMemoryFact:
    fact: str
    source_conversation_id: str
    extracted_at: str
    confidence: float


def normalize_fact(fact: str) -> str:
    """Normalize a fact for dedupe: strip all whitespace, full-width space -> half, lowercase."""
    return "".join(fact.split()).replace("\u3000", " ").lower()


_FACT_CATEGORY_PREFIXES = ("用户称呼", "用户地址", "用户孩子", "用户手机号")

# (规范前缀, 宽松关键词)：LLM 提取的文本表述不固定（如"用户的称呼是小明"），
# 只要包含类别关键词就归入同一去重桶，避免规则与 LLM 提取互相重复。
_FACT_CATEGORY_KEYWORDS = (
    ("用户称呼", "称呼"),
    ("用户称呼", "名字"),
    ("用户地址", "地址"),
    ("用户孩子", "孩子"),
    ("用户手机号", "手机号"),
)


def fact_category_key(fact: str) -> str:
    """Dedupe key: same fact category keeps only the latest value."""
    for prefix, keyword in _FACT_CATEGORY_KEYWORDS:
        if keyword in fact or prefix in fact:
            return prefix
    return normalize_fact(fact)


def _rule_extract(message: str, conversation_id: str) -> list[UserMemoryFact]:
    now = datetime.now(timezone.utc).isoformat()
    facts: list[UserMemoryFact] = []
    for pattern, template in _FACT_PATTERNS:
        for match in pattern.finditer(message):
            facts.append(
                UserMemoryFact(
                    fact=template.format(match.group(1)),
                    source_conversation_id=conversation_id,
                    extracted_at=now,
                    confidence=0.8,
                )
            )
    return facts


def extract_user_facts(message: str, conversation_id: str) -> list[UserMemoryFact]:
    """Rule-based extraction; LLM extraction is layered on separately."""
    return _rule_extract(message, conversation_id)


def extract_user_facts_llm(
    message: str,
    conversation_id: str,
    settings,
    *,
    client=None,
) -> list[UserMemoryFact]:
    """LLM-assisted extraction; returns [] without a key or on any failure."""
    if not getattr(settings, "has_llm_api_key", False):
        return []
    try:
        from app.services.llm_service import LLMChatService

        llm = client or LLMChatService(settings)
        raw = llm.generate_reply(
            (
                "从用户消息中提取可长期记住的事实（称呼/地址/家庭/联系方式）。"
                '只输出 JSON：{"facts": [{"text": str, "confidence": float}]}；无事实输出 {"facts": []}。\n'
                '每条 text 必须以固定前缀开头："用户称呼：" / "用户地址：" / "用户孩子：" / "用户手机号："。\n\n'
                f"用户消息：{message}"
            )
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        now = datetime.now(timezone.utc).isoformat()
        return [
            UserMemoryFact(
                fact=str(item["text"]),
                source_conversation_id=conversation_id,
                extracted_at=now,
                confidence=float(item.get("confidence", 0.9)),
            )
            for item in data.get("facts", [])
        ]
    except Exception:
        logger.warning("user_memory_llm_extract_failed conversation_id=%s", conversation_id)
        return []


class MemoryService:
    """Redis persistence for user memory; failures are silent (memory is optional)."""

    def __init__(self, settings=None, client=None) -> None:
        self._settings = settings
        self._client = client

    def _key(self, tenant_id: str, user_id: str) -> str:
        return f"ai-service:user-memory:{tenant_id}:{user_id}"

    def _redis(self):
        if self._client is not None:
            return self._client
        if self._settings is None:
            raise RuntimeError("MemoryService needs settings or client")
        return redis_lib.Redis.from_url(self._settings.resolved_agent_redis_url)

    def get(self, tenant_id: str, user_id: str) -> list[dict]:
        try:
            raw = self._redis().get(self._key(tenant_id, user_id))
        except Exception:
            logger.warning("user_memory_read_failed tenant=%s user=%s", tenant_id, user_id)
            return []
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return []

    def put(self, tenant_id: str, user_id: str, items: list[dict], ttl_seconds: int) -> None:
        try:
            r = self._redis()
            r.set(
                self._key(tenant_id, user_id),
                json.dumps(items, ensure_ascii=False),
                ex=ttl_seconds,
            )
        except Exception:
            logger.warning("user_memory_write_failed tenant=%s user=%s", tenant_id, user_id)

    def clear(self, tenant_id: str, user_id: str) -> None:
        try:
            self._redis().delete(self._key(tenant_id, user_id))
        except Exception:
            logger.warning("user_memory_clear_failed tenant=%s user=%s", tenant_id, user_id)


class UserMemoryStore:
    def __init__(
        self,
        *,
        memory_service: MemoryService | None = None,
        ttl_seconds: int = 43200,
        settings=None,
    ) -> None:
        self._memory = memory_service or MemoryService(settings=settings)
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _to_dict(fact: UserMemoryFact) -> dict:
        return {
            "fact": fact.fact,
            "source_conversation_id": fact.source_conversation_id,
            "extracted_at": fact.extracted_at,
            "confidence": fact.confidence,
        }

    def get_facts(self, actor) -> list[dict]:
        return self._memory.get(actor.tenant_id, actor.user_id)

    def store_facts(self, actor, facts: list[UserMemoryFact]) -> None:
        existing = self._memory.get(actor.tenant_id, actor.user_id)
        by_key = {fact_category_key(item["fact"]): item for item in existing}
        for fact in facts:
            by_key[fact_category_key(fact.fact)] = self._to_dict(fact)
        self._memory.put(
            actor.tenant_id,
            actor.user_id,
            list(by_key.values()),
            self._ttl_seconds,
        )

    def clear(self, actor) -> None:
        self._memory.clear(actor.tenant_id, actor.user_id)
