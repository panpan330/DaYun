"""情绪分类器测试：rule_based / fake_llm / real_llm 三模式。"""

import pytest

from app.agents.emotion_classifier import (
    CustomerEmotion,
    EMOTION_HANDOFF_EMOTIONS,
    FakeLLMEmotionClassifier,
    LLMEmotionClassifier,
    RuleBasedEmotionClassifier,
)


class TestRuleBased:
    def test_angry_keywords(self):
        result = RuleBasedEmotionClassifier().recognize("你们太气人了，这是骗子！我要投诉！")
        assert result.emotion == CustomerEmotion.ANGRY

    def test_anxious_keywords(self):
        result = RuleBasedEmotionClassifier().recognize("我特别着急，货还没到怎么办")
        assert result.emotion == CustomerEmotion.ANXIOUS

    def test_urgent_keywords(self):
        result = RuleBasedEmotionClassifier().recognize("请快点处理，立刻马上！")
        assert result.emotion == CustomerEmotion.URGENT

    def test_satisfied_keywords(self):
        result = RuleBasedEmotionClassifier().recognize("谢谢，处理得很满意！")
        assert result.emotion == CustomerEmotion.SATISFIED

    def test_apologetic_keywords(self):
        result = RuleBasedEmotionClassifier().recognize("不好意思，麻烦你们了，抱歉")
        assert result.emotion == CustomerEmotion.APOLOGETIC

    def test_neutral_when_no_keyword(self):
        result = RuleBasedEmotionClassifier().recognize("退款多久到账？")
        assert result.emotion == CustomerEmotion.NEUTRAL

    def test_angry_priority_over_urgent(self):
        result = RuleBasedEmotionClassifier().recognize("气死我了！快点处理！")
        assert result.emotion == CustomerEmotion.ANGRY

    def test_reason_is_filled(self):
        result = RuleBasedEmotionClassifier().recognize("你们太气人了")
        assert result.reason


class TestFakeLLM:
    def test_deterministic(self):
        result = FakeLLMEmotionClassifier().recognize("退款多久到账？")
        assert result.emotion == CustomerEmotion.NEUTRAL


class TestLLM:
    def test_invalid_output_falls_back_neutral(self):
        class BadClient:
            def chat_completions_create(self, **kwargs):
                return {"choices": [{"message": {"content": '{"emotion": "not-a-valid-emotion"}'}}]}

        from app.core.config import Settings

        classifier = LLMEmotionClassifier.__new__(LLMEmotionClassifier)
        classifier.settings = Settings(_env_file=None)
        classifier._client = BadClient()
        result = classifier.recognize("随便一句话")
        assert result.emotion == CustomerEmotion.NEUTRAL


def test_handoff_emotion_set():
    assert CustomerEmotion.ANGRY in EMOTION_HANDOFF_EMOTIONS
    assert CustomerEmotion.ANXIOUS in EMOTION_HANDOFF_EMOTIONS
    assert CustomerEmotion.URGENT in EMOTION_HANDOFF_EMOTIONS
    assert CustomerEmotion.DISSATISFIED not in EMOTION_HANDOFF_EMOTIONS
