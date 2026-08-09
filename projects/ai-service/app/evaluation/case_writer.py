from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any

from app.agents.intent_evaluation import (
    AgentEvalCase,
    AgentEvalDataset,
    AgentEvalInputs,
    AgentEvalMetadata,
    load_agent_eval_dataset,
)
from app.agents.ticket_agent import TICKET_AGENT_INTENT_ROUTES
from app.evaluation.bad_case_registry import BadCaseRecord


_agent_cases_lock = Lock()


def _build_expected_from_bad_case(record: BadCaseRecord) -> dict[str, Any]:
    """按 failure_layer / 断言类型生成 expected 字典（intent_route 取自
    TICKET_AGENT_INTENT_ROUTES，满足 AgentEvalExpected 的 route 匹配校验）。

    - intent 断言 → {intent: production spec.expected_intent, intent_route}
    - security / must_not_reveal → {intent: unsupported, intent_route, must_not_reveal}
    - 其它（tool_called / must_ask_for / citation 等）→ 保守默认
      {intent: unclear, intent_route: ask_clarifying_question}
    """
    spec = record.production_regression
    if spec is not None and spec.assertion == "intent" and spec.expected_intent is not None:
        intent = spec.expected_intent
        return {
            "intent": intent,
            "intent_route": TICKET_AGENT_INTENT_ROUTES[intent],
        }
    if spec is not None and spec.assertion == "must_not_reveal":
        return {
            "intent": "unsupported",
            "intent_route": TICKET_AGENT_INTENT_ROUTES["unsupported"],
            "must_not_reveal": list(spec.must_not_reveal_terms),
        }
    if record.failure_layer == "security":
        terms = list(spec.must_not_reveal_terms) if spec is not None else []
        return {
            "intent": "unsupported",
            "intent_route": TICKET_AGENT_INTENT_ROUTES["unsupported"],
            "must_not_reveal": terms,
        }
    return {
        "intent": "unclear",
        "intent_route": TICKET_AGENT_INTENT_ROUTES["unclear"],
    }


def write_bad_case_to_agent_cases(
    record: BadCaseRecord,
    *,
    cases_path: Path,
) -> AgentEvalCase | None:
    """把一个已 promote 的 bad case 写回 agent eval 正式用例集（agent_cases.json）。

    幂等：已存在同 id / 同 source_bad_case_id（编码在 tags）/ 同 message 的用例时
    返回 None。写回保持 schema_version/description 不变，原子写（.tmp + replace）。
    """
    path = Path(cases_path)
    new_case = _build_agent_eval_case(record)
    with _agent_cases_lock:
        dataset = load_agent_eval_dataset(path)
        if _has_matching_case(dataset, record, new_case):
            return None
        updated = dataset.model_copy(update={"cases": [*dataset.cases, new_case]})
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(updated.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return new_case


def _build_agent_eval_case(record: BadCaseRecord) -> AgentEvalCase:
    spec = record.production_regression
    message = (spec.message if spec is not None else "") or record.evidence_summary
    case_id = _slugify(f"prod_{record.id}_regression")
    return AgentEvalCase(
        id=case_id,
        name=f"Regression: {record.title}",
        inputs=AgentEvalInputs(message=message),
        expected=_build_expected_from_bad_case(record),
        metadata=AgentEvalMetadata(
            task_type=record.task_type,
            business_domain="production",
            case_type="production_regression",
            difficulty="hard",
            priority="p0",
            tags=_unique_strings(
                [
                    "regression",
                    "from_bad_case",
                    record.failure_layer,
                    # AgentEvalMetadata 无 source_bad_case_id 字段，编码进 tags 以支撑幂等去重。
                    f"source_bad_case_id:{record.id}",
                ]
            ),
        ),
    )


def _has_matching_case(
    dataset: AgentEvalDataset,
    record: BadCaseRecord,
    new_case: AgentEvalCase,
) -> bool:
    source_tag = f"source_bad_case_id:{record.id}"
    message = new_case.inputs.message
    for case in dataset.cases:
        if case.id == new_case.id:
            return True
        if source_tag in case.metadata.tags:
            return True
        if case.inputs.message == message:
            return True
    return False


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "bad_case"


def _unique_strings(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)
    return unique_values
