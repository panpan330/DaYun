from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from app.core.exceptions import AppException


AISecuritySource = Literal["user", "history", "rag", "tool_result", "model_output"]
AISecurityReason = Literal[
    "allowed",
    "prompt_injection_detected",
    "sensitive_output_detected",
]

PROMPT_INJECTION_REJECTION_MESSAGE = (
    "请求包含疑似提示词注入或越权指令，系统已拒绝处理。"
)


@dataclass(frozen=True)
class SecuritySignalRule:
    code: str
    pattern: re.Pattern[str]
    description: str


@dataclass(frozen=True)
class AISecurityDecision:
    allowed: bool
    reason: AISecurityReason
    source: AISecuritySource
    matched_code: str | None = None
    safe_message: str | None = None

    def to_log_fields(self) -> dict[str, str | bool]:
        fields: dict[str, str | bool] = {
            "ai_security.allowed": self.allowed,
            "ai_security.reason": self.reason,
            "ai_security.source": self.source,
        }
        if self.matched_code is not None:
            fields["ai_security.matched_code"] = self.matched_code
        return fields


PROMPT_INJECTION_RULES: tuple[SecuritySignalRule, ...] = (
    SecuritySignalRule(
        "PROMPT_INJECTION_IGNORE_INSTRUCTIONS",
        re.compile(
            r"\b(ignore|disregard)\s+(all\s+)?(previous|prior|above|system|developer)"
            r"\s+(instructions|rules|messages)\b",
            re.IGNORECASE,
        ),
        "User text asks the model to ignore higher-priority instructions.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_REVEAL_SYSTEM_PROMPT",
        re.compile(
            r"\b(reveal|show|print|dump|leak)\b.{0,40}"
            r"\b(system prompt|developer message|hidden instructions)\b",
            re.IGNORECASE,
        ),
        "User text asks the model to reveal hidden instructions.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_BYPASS_SECURITY",
        re.compile(
            r"\b(bypass|disable|override)\b.{0,40}"
            r"\b(safety|security|permission|authorization|policy)\b",
            re.IGNORECASE,
        ),
        "User text asks the model to bypass safety or permission boundaries.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_UNAUTHORIZED_TOOL",
        re.compile(
            r"\b(call|execute|invoke|use)\b.{0,40}"
            r"\b(hidden|internal|unauthorized|forbidden)\b.{0,20}\b(tool|function|api)\b",
            re.IGNORECASE,
        ),
        "User text asks the model to call unauthorized internal tools.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_LEAK_SECRET",
        re.compile(
            r"\b(print|show|reveal|leak|dump)\b.{0,40}"
            r"\b(api key|secret|token|password|credential)\b",
            re.IGNORECASE,
        ),
        "User text asks the model to leak credentials or secrets.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_IGNORE_INSTRUCTIONS_CN",
        re.compile(r"(忽略|无视).{0,16}(以上|上面|之前|系统|开发者).{0,16}(指令|提示|规则)"),
        "User text asks the model to ignore Chinese instructions.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_REVEAL_SYSTEM_PROMPT_CN",
        re.compile(r"(输出|泄露|展示|打印|告诉我).{0,20}(系统提示词|开发者消息|隐藏指令|内部规则)"),
        "User text asks the model to reveal Chinese hidden instructions.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_BYPASS_SECURITY_CN",
        re.compile(r"(绕过|无视|禁用).{0,20}(权限|安全|限制|审核|校验)"),
        "User text asks the model to bypass Chinese safety or permission boundaries.",
    ),
    SecuritySignalRule(
        "PROMPT_INJECTION_UNAUTHORIZED_TOOL_CN",
        re.compile(r"(调用|使用|执行).{0,20}(隐藏|内部|未授权|禁止).{0,12}(工具|函数|接口)"),
        "User text asks the model to call unauthorized Chinese internal tools.",
    ),
)

SENSITIVE_TEXT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"sk-[A-Za-z0-9._-]{12,}"),
        "[REDACTED_API_KEY]",
    ),
    (
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{12,}"),
        "Bearer [REDACTED_TOKEN]",
    ),
    (
        re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
)


def inspect_prompt_injection(
    text: str,
    *,
    source: AISecuritySource,
) -> AISecurityDecision:
    normalized_text = text.strip()
    if not normalized_text:
        return AISecurityDecision(
            allowed=True,
            reason="allowed",
            source=source,
        )

    for rule in PROMPT_INJECTION_RULES:
        if rule.pattern.search(normalized_text):
            return AISecurityDecision(
                allowed=False,
                reason="prompt_injection_detected",
                source=source,
                matched_code=rule.code,
                safe_message=PROMPT_INJECTION_REJECTION_MESSAGE,
            )

    return AISecurityDecision(
        allowed=True,
        reason="allowed",
        source=source,
    )


def require_prompt_injection_safe(
    text: str,
    *,
    source: AISecuritySource,
) -> None:
    decision = inspect_prompt_injection(text, source=source)
    if decision.allowed:
        return
    raise AppException(
        code="PROMPT_INJECTION_DETECTED",
        message=decision.safe_message or PROMPT_INJECTION_REJECTION_MESSAGE,
        status_code=400,
    )


def require_texts_prompt_injection_safe(
    texts: Iterable[tuple[str, AISecuritySource]],
) -> None:
    for text, source in texts:
        require_prompt_injection_safe(text, source=source)


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern, replacement in SENSITIVE_TEXT_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def inspect_sensitive_output(text: str) -> AISecurityDecision:
    redacted = redact_sensitive_text(text)
    if redacted == text:
        return AISecurityDecision(
            allowed=True,
            reason="allowed",
            source="model_output",
        )
    return AISecurityDecision(
        allowed=False,
        reason="sensitive_output_detected",
        source="model_output",
        matched_code="SENSITIVE_OUTPUT_REDACTED",
        safe_message="模型输出包含敏感信息，已进行脱敏处理。",
    )
