"""情绪识别分类器（rule_based / fake_llm / real_llm 三模式，复用 intent 分类模式）。"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import Settings


class CustomerEmotion(str, Enum):
    ANGRY = "angry"
    ANXIOUS = "anxious"
    DISSATISFIED = "dissatisfied"
    URGENT = "urgent"
    APOLOGETIC = "apologetic"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


EMOTION_HANDOFF_EMOTIONS = frozenset(
    {CustomerEmotion.ANGRY, CustomerEmotion.ANXIOUS, CustomerEmotion.URGENT}
)


class CustomerEmotionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion: CustomerEmotion
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class CustomerEmotionClassifier(Protocol):
    def recognize(self, message: str) -> CustomerEmotionResult:
        """Return a validated emotion classification for the user message."""


_EMOTION_KEYWORDS: dict[CustomerEmotion, tuple[str, ...]] = {
    CustomerEmotion.ANGRY: (
        "气死",
        "气人",
        "骗子",
        "太过分",
        "别太过分",
        "不要太过分",
        "愤怒",
        "凭什么",
        "无法接受",
        "忍无可忍",
        "垃圾",
    ),
    CustomerEmotion.URGENT: (
        "快点",
        "立刻",
        "马上",
        "赶紧",
        "尽快",
        "现在就",
        "立刻马上",
        "加急",
        "十万火急",
    ),
    CustomerEmotion.ANXIOUS: (
        "着急",
        "焦虑",
        "担心",
        "还没到",
        "等好久",
        "怕",
        "慌",
    ),
    CustomerEmotion.DISSATISFIED: (
        "不满意",
        "失望",
        "态度差",
        "敷衍",
        "无语",
        "唉",
        "太慢了",
    ),
    CustomerEmotion.APOLOGETIC: (
        "抱歉",
        "不好意思",
        "对不起",
        "麻烦了",
        "打扰了",
        "添麻烦",
    ),
    CustomerEmotion.SATISFIED: (
        "谢谢",
        "满意",
        "太好了",
        "不错",
        "给力",
        "感谢",
        "赞",
    ),
}

# 命中优先级：angry > urgent > anxious > dissatisfied > apologetic > satisfied
_EMOTION_PRIORITY = (
    CustomerEmotion.ANGRY,
    CustomerEmotion.URGENT,
    CustomerEmotion.ANXIOUS,
    CustomerEmotion.DISSATISFIED,
    CustomerEmotion.APOLOGETIC,
    CustomerEmotion.SATISFIED,
)


def classify_customer_emotion(message: str) -> CustomerEmotionResult:
    """基于关键词的确定性情绪分类（rule_based 核心）。"""
    for emotion in _EMOTION_PRIORITY:
        for keyword in _EMOTION_KEYWORDS[emotion]:
            if keyword in message:
                return CustomerEmotionResult(
                    emotion=emotion,
                    reason=f"命中关键词「{keyword}」",
                )
    return CustomerEmotionResult(
        emotion=CustomerEmotion.NEUTRAL,
        reason="未命中情绪关键词",
    )


class RuleBasedEmotionClassifier:
    def recognize(self, message: str) -> CustomerEmotionResult:
        return classify_customer_emotion(message)


class FakeLLMEmotionClassifier:
    def recognize(self, message: str) -> CustomerEmotionResult:
        result = classify_customer_emotion(message)
        raw_json = json.dumps(result.model_dump(), ensure_ascii=False)
        return parse_customer_emotion_result_json(raw_json)


def parse_customer_emotion_result_json(raw_json: str) -> CustomerEmotionResult:
    """解析 LLM JSON 输出为校验后的结果；非法值抛 ValidationError。"""
    data = json.loads(raw_json)
    return CustomerEmotionResult.model_validate(data)


class LLMEmotionClassifier:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def recognize(self, message: str) -> CustomerEmotionResult:
        if not self.settings.has_llm_api_key:
            return CustomerEmotionResult(
                emotion=CustomerEmotion.NEUTRAL,
                reason="LLM API key 未配置，回退为中性",
            )
        allowed = "|".join(emotion.value for emotion in CustomerEmotion)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是客服情绪识别器。只输出 JSON："
                    f'{{"emotion": <{allowed}>, "reason": "简短中文理由"}}'
                    "emotion 只能是列表中的值。"
                ),
            },
            {"role": "user", "content": message},
        ]
        try:
            from app.services.llm_client import create_openai_compatible_client

            client = self._client or create_openai_compatible_client(self.settings)
            response = client.chat_completions_create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=0,
            )
            content = response["choices"][0]["message"]["content"]
            return parse_customer_emotion_result_json(content)
        except Exception as exc:  # 校验失败/调用失败 → 兜底 neutral
            return CustomerEmotionResult(
                emotion=CustomerEmotion.NEUTRAL,
                reason=f"情绪识别失败，回退为中性（{type(exc).__name__}）",
            )


def build_emotion_classifier(
    mode: str,
    settings: Settings | None = None,
    client: Any | None = None,
) -> CustomerEmotionClassifier:
    if mode == "fake_llm":
        return FakeLLMEmotionClassifier()
    if mode == "real_llm":
        if settings is None:
            raise ValueError("real_llm 需要 settings")
        return LLMEmotionClassifier(settings, client=client)
    return RuleBasedEmotionClassifier()
