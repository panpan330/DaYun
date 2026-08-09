from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.documents import RetrievedChunk


class RagSecurityFindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RagSecurityFindingCategory(str, Enum):
    PERMISSION = "permission"
    PROMPT_INJECTION = "prompt_injection"
    SENSITIVE_DATA = "sensitive_data"


class RagSecurityRiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class PromptInjectionRule:
    pattern: str
    code: str
    message: str
    severity: RagSecurityFindingSeverity


class RagSecurityPolicy(BaseModel):
    allowed_permission_groups: list[str] | None = Field(default=None)
    block_on_prompt_injection: bool = True
    block_on_sensitive_data: bool = True
    scan_metadata_for_prompt_injection: bool = True
    prompt_injection_blocking_severities: list[RagSecurityFindingSeverity] = Field(
        default_factory=lambda: [
            RagSecurityFindingSeverity.HIGH,
            RagSecurityFindingSeverity.CRITICAL,
        ]
    )

    @field_validator("allowed_permission_groups", mode="before")
    @classmethod
    def reject_invalid_permission_groups(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("allowed_permission_groups must be a list of strings")
        groups: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("allowed_permission_groups must contain non-blank strings")
            groups.append(item.strip())
        return groups

    @model_validator(mode="after")
    def reject_empty_prompt_injection_blocking_severities(self) -> "RagSecurityPolicy":
        if self.block_on_prompt_injection and not self.prompt_injection_blocking_severities:
            raise ValueError("prompt_injection_blocking_severities must not be empty")
        return self


class RagSecurityFinding(BaseModel):
    code: str = Field(min_length=1)
    category: RagSecurityFindingCategory
    severity: RagSecurityFindingSeverity
    message: str = Field(min_length=1)
    chunk_id: str | None = None
    source: str | None = None
    field: str | None = None
    evidence: str | None = None


class RagSecurityReport(BaseModel):
    query: str = Field(min_length=1)
    risk_level: RagSecurityRiskLevel = RagSecurityRiskLevel.SAFE
    checked_chunk_count: int = Field(ge=0)
    safe_chunk_count: int = Field(ge=0)
    blocked_chunk_count: int = Field(ge=0)
    safe_chunks: list[RetrievedChunk] = Field(default_factory=list)
    blocked_chunk_ids: list[str] = Field(default_factory=list)
    findings: list[RagSecurityFinding] = Field(default_factory=list)
    finding_count_by_category: dict[str, int] = Field(default_factory=dict)
    blocked_reason_codes: list[str] = Field(default_factory=list)


PROMPT_INJECTION_RULES: tuple[PromptInjectionRule, ...] = (
    PromptInjectionRule(
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        "RAG_PROMPT_INJECTION_IGNORE_INSTRUCTIONS",
        "Document text appears to ask the model to ignore existing instructions.",
        RagSecurityFindingSeverity.HIGH,
    ),
    PromptInjectionRule(
        r"(?i)\breveal\s+(the\s+)?(system|developer)\s+(prompt|message|instructions)\b",
        "RAG_PROMPT_INJECTION_REVEAL_SYSTEM_PROMPT",
        "Document text appears to ask the model to reveal system or developer instructions.",
        RagSecurityFindingSeverity.CRITICAL,
    ),
    PromptInjectionRule(
        r"(?i)\byou\s+are\s+now\s+(not\s+)?(a\s+)?(different|unrestricted|developer)\b",
        "RAG_PROMPT_INJECTION_ROLE_OVERRIDE",
        "Document text appears to override the model role or safety boundary.",
        RagSecurityFindingSeverity.HIGH,
    ),
    PromptInjectionRule(
        r"(?i)\b(call|execute|invoke|use)\s+(the\s+)?(tool|function|api)\b",
        "RAG_PROMPT_INJECTION_TOOL_ABUSE",
        "Document text appears to instruct the model to call tools or APIs.",
        RagSecurityFindingSeverity.HIGH,
    ),
    PromptInjectionRule(
        r"(?i)(^|\n)\s*(system|developer|assistant)\s*:",
        "RAG_PROMPT_INJECTION_ROLE_DELIMITER",
        "Document text contains role-like delimiters that may confuse prompt boundaries.",
        RagSecurityFindingSeverity.MEDIUM,
    ),
    PromptInjectionRule(
        r"(?i)`{3,}\s*(system|developer|assistant)|<\s*/?\s*(system|developer|assistant)\s*>",
        "RAG_PROMPT_INJECTION_MARKUP_DELIMITER",
        "Document text contains markup-like delimiters that may imitate prompt roles.",
        RagSecurityFindingSeverity.MEDIUM,
    ),
    PromptInjectionRule(
        r"(?i)\b(do\s+not|don't)\s+(cite|use\s+sources|follow\s+(the\s+)?output\s+schema)\b",
        "RAG_PROMPT_INJECTION_OUTPUT_POLICY_OVERRIDE",
        "Document text appears to override citation or output-format rules.",
        RagSecurityFindingSeverity.MEDIUM,
    ),
    PromptInjectionRule(
        r"(忽略|无视).{0,16}(以上|上面|之前|系统|开发者).{0,16}(指令|提示|规则)",
        "RAG_PROMPT_INJECTION_IGNORE_CN",
        "Document text appears to ask the model to ignore existing Chinese instructions.",
        RagSecurityFindingSeverity.HIGH,
    ),
    PromptInjectionRule(
        r"(泄露|输出|展示|打印).{0,16}(系统提示词|开发者消息|系统指令|内部规则)",
        "RAG_PROMPT_INJECTION_EXFILTRATE_CN",
        "Document text appears to ask the model to leak internal prompts or rules.",
        RagSecurityFindingSeverity.CRITICAL,
    ),
)

SENSITIVE_DATA_RULES: tuple[tuple[str, str, str, RagSecurityFindingSeverity], ...] = (
    (
        r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|secret)\b\s*[:=]\s*['\"]?[^'\"\s]{8,}",
        "RAG_SENSITIVE_CREDENTIAL",
        "Document text appears to contain credentials, secrets, or access tokens.",
        RagSecurityFindingSeverity.CRITICAL,
    ),
    (
        r"(?i)\bauthorization\s*:\s*bearer\s+[a-z0-9._-]{8,}",
        "RAG_SENSITIVE_BEARER_TOKEN",
        "Document text appears to contain a Bearer token.",
        RagSecurityFindingSeverity.CRITICAL,
    ),
    (
        r"(?i)-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
        "RAG_SENSITIVE_PRIVATE_KEY",
        "Document text appears to contain a private key.",
        RagSecurityFindingSeverity.CRITICAL,
    ),
    (
        r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "RAG_SENSITIVE_PHONE_NUMBER",
        "Document text appears to contain a phone number.",
        RagSecurityFindingSeverity.HIGH,
    ),
    (
        r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
        "RAG_SENSITIVE_EMAIL",
        "Document text appears to contain an email address.",
        RagSecurityFindingSeverity.MEDIUM,
    ),
)


def inspect_retrieved_chunks(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    policy: RagSecurityPolicy | None = None,
) -> RagSecurityReport:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")

    active_policy = policy or RagSecurityPolicy()
    safe_chunks: list[RetrievedChunk] = []
    blocked_chunk_ids: list[str] = []
    findings: list[RagSecurityFinding] = []

    for chunk in chunks:
        chunk_findings = inspect_chunk_security(chunk, policy=active_policy)
        findings.extend(chunk_findings)
        if any(_is_blocking_finding(finding, active_policy) for finding in chunk_findings):
            blocked_chunk_ids.append(chunk.chunk_id)
            continue
        safe_chunks.append(chunk)

    return RagSecurityReport(
        query=normalized_query,
        risk_level=_build_risk_level(findings, blocked_chunk_ids, policy=active_policy),
        checked_chunk_count=len(chunks),
        safe_chunk_count=len(safe_chunks),
        blocked_chunk_count=len(blocked_chunk_ids),
        safe_chunks=safe_chunks,
        blocked_chunk_ids=blocked_chunk_ids,
        findings=findings,
        finding_count_by_category=_count_findings_by_category(findings),
        blocked_reason_codes=_blocked_reason_codes(findings, policy=active_policy),
    )


def inspect_chunk_security(
    chunk: RetrievedChunk,
    *,
    policy: RagSecurityPolicy | None = None,
) -> list[RagSecurityFinding]:
    active_policy = policy or RagSecurityPolicy()
    findings: list[RagSecurityFinding] = []
    findings.extend(_inspect_permission(chunk, active_policy))
    findings.extend(_inspect_prompt_injection(chunk, active_policy))
    findings.extend(_inspect_sensitive_data(chunk))
    return findings


def format_security_report_for_debug(report: RagSecurityReport) -> list[str]:
    lines = [
        (
            f"checked={report.checked_chunk_count} safe={report.safe_chunk_count} "
            f"blocked={report.blocked_chunk_count} findings={len(report.findings)}"
        )
    ]
    for finding in report.findings:
        source = finding.source or "unknown-source"
        chunk_id = finding.chunk_id or "unknown-chunk"
        field = finding.field or "unknown-field"
        evidence = f" evidence={finding.evidence}" if finding.evidence else ""
        lines.append(
            (
                f"{finding.severity.value} {finding.category.value} "
                f"code={finding.code} source={source} chunk_id={chunk_id} "
                f"field={field}{evidence}"
            )
        )
    return lines


def _inspect_permission(
    chunk: RetrievedChunk,
    policy: RagSecurityPolicy,
) -> list[RagSecurityFinding]:
    if policy.allowed_permission_groups is None:
        return []

    permission_group = chunk.metadata.get("permission_group")
    if not isinstance(permission_group, str) or not permission_group.strip():
        return [
            _finding(
                chunk,
                code="RAG_PERMISSION_GROUP_MISSING",
                category=RagSecurityFindingCategory.PERMISSION,
                severity=RagSecurityFindingSeverity.CRITICAL,
                message="Retrieved chunk is missing permission_group.",
                field="metadata.permission_group",
            )
        ]

    if permission_group.strip() not in policy.allowed_permission_groups:
        return [
            _finding(
                chunk,
                code="RAG_PERMISSION_GROUP_DENIED",
                category=RagSecurityFindingCategory.PERMISSION,
                severity=RagSecurityFindingSeverity.CRITICAL,
                message="Retrieved chunk permission_group is outside the allowed range.",
                field="metadata.permission_group",
                evidence=permission_group.strip(),
            )
        ]
    return []


def _inspect_prompt_injection(
    chunk: RetrievedChunk,
    policy: RagSecurityPolicy,
) -> list[RagSecurityFinding]:
    findings: list[RagSecurityFinding] = []
    for field_name, text in _iter_prompt_injection_text_fields(chunk, policy):
        for rule in PROMPT_INJECTION_RULES:
            for match in re.finditer(rule.pattern, text):
                findings.append(
                    _finding(
                        chunk,
                        code=rule.code,
                        category=RagSecurityFindingCategory.PROMPT_INJECTION,
                        severity=rule.severity,
                        message=rule.message,
                        field=field_name,
                        evidence=_clip_evidence(match.group(0)),
                    )
                )
    return findings


def _inspect_sensitive_data(chunk: RetrievedChunk) -> list[RagSecurityFinding]:
    findings: list[RagSecurityFinding] = []
    for field_name, text in _iter_scanned_text_fields(chunk):
        for pattern, code, message, severity in SENSITIVE_DATA_RULES:
            for match in re.finditer(pattern, text):
                findings.append(
                    _finding(
                        chunk,
                        code=code,
                        category=RagSecurityFindingCategory.SENSITIVE_DATA,
                        severity=severity,
                        message=message,
                        field=field_name,
                        evidence=_redacted_evidence(match.group(0)),
                    )
                )
    return findings


def _iter_prompt_injection_text_fields(
    chunk: RetrievedChunk,
    policy: RagSecurityPolicy,
) -> Iterable[tuple[str, str]]:
    yield "content", chunk.content
    if not policy.scan_metadata_for_prompt_injection:
        return
    for key in ("source", "title", "section", "file_name"):
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            yield f"metadata.{key}", value


def _iter_scanned_text_fields(chunk: RetrievedChunk) -> Iterable[tuple[str, str]]:
    yield "content", chunk.content
    for key in ("source", "title", "section"):
        value = chunk.metadata.get(key)
        if isinstance(value, str):
            yield f"metadata.{key}", value


def _is_blocking_finding(
    finding: RagSecurityFinding,
    policy: RagSecurityPolicy,
) -> bool:
    if finding.category is RagSecurityFindingCategory.PERMISSION:
        return True
    if (
        finding.category is RagSecurityFindingCategory.PROMPT_INJECTION
        and policy.block_on_prompt_injection
    ):
        return finding.severity in set(policy.prompt_injection_blocking_severities)
    if (
        finding.category is RagSecurityFindingCategory.SENSITIVE_DATA
        and policy.block_on_sensitive_data
    ):
        return finding.severity in {
            RagSecurityFindingSeverity.HIGH,
            RagSecurityFindingSeverity.CRITICAL,
        }
    return False


def _finding(
    chunk: RetrievedChunk,
    *,
    code: str,
    category: RagSecurityFindingCategory,
    severity: RagSecurityFindingSeverity,
    message: str,
    field: str,
    evidence: str | None = None,
) -> RagSecurityFinding:
    source = chunk.metadata.get("source")
    return RagSecurityFinding(
        code=code,
        category=category,
        severity=severity,
        message=message,
        chunk_id=chunk.chunk_id,
        source=source if isinstance(source, str) and source.strip() else None,
        field=field,
        evidence=evidence,
    )


def _build_risk_level(
    findings: Sequence[RagSecurityFinding],
    blocked_chunk_ids: Sequence[str],
    *,
    policy: RagSecurityPolicy,
) -> RagSecurityRiskLevel:
    if blocked_chunk_ids or any(_is_blocking_finding(finding, policy) for finding in findings):
        return RagSecurityRiskLevel.BLOCKED
    if findings:
        return RagSecurityRiskLevel.WARNING
    return RagSecurityRiskLevel.SAFE


def _count_findings_by_category(
    findings: Sequence[RagSecurityFinding],
) -> dict[str, int]:
    return dict(Counter(finding.category.value for finding in findings))


def _blocked_reason_codes(
    findings: Sequence[RagSecurityFinding],
    *,
    policy: RagSecurityPolicy,
) -> list[str]:
    codes: list[str] = []
    for finding in findings:
        if _is_blocking_finding(finding, policy) and finding.code not in codes:
            codes.append(finding.code)
    return codes


def _clip_evidence(value: str, *, max_length: int = 80) -> str:
    compact = " ".join(value.strip().split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


def _redacted_evidence(_: str) -> str:
    return "[redacted]"
