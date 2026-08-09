from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PromptParts:
    task: str
    context: str | None = None
    constraints: Sequence[str] = ()
    output_format: str | None = None
    failure_policy: str | None = None


def _require_text(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank")
    return stripped


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _clean_constraints(constraints: Sequence[str]) -> list[str]:
    return [constraint.strip() for constraint in constraints if constraint.strip()]


def build_clear_user_prompt(parts: PromptParts) -> str:
    sections = [f"## 任务\n{_require_text(parts.task, 'task')}"]

    context = _optional_text(parts.context)
    if context is not None:
        sections.append(f"## 上下文\n{context}")

    constraints = _clean_constraints(parts.constraints)
    if constraints:
        constraint_lines = "\n".join(f"- {constraint}" for constraint in constraints)
        sections.append(f"## 要求\n{constraint_lines}")

    output_format = _optional_text(parts.output_format)
    if output_format is not None:
        sections.append(f"## 输出格式\n{output_format}")

    failure_policy = _optional_text(parts.failure_policy)
    if failure_policy is not None:
        sections.append(f"## 无法完成时\n{failure_policy}")

    return "\n\n".join(sections)
