"""Shared assertions for agent evaluation cases.

Task 1 of the eval-deepening effort: enforce the previously dead
``must_ask_for`` / ``must_not_reveal`` fields of :class:`AgentEvalExpected`.

``check_must_ask_for`` returns the requested fields whose aliases do NOT
appear in the reply (i.e. the agent failed to ask for them); a non-empty
result means the case failed. ``check_must_not_reveal`` returns the secret
terms that DO appear in the reply; a non-empty result means the case failed.

Both functions are safe on missing/empty replies: an empty reply fails
``must_ask_for`` (fail closed - the agent did not ask) and passes
``must_not_reveal`` (nothing can leak from an empty answer).
"""

MUST_ASK_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": ("order_id", "订单号"),
    "description": ("description", "描述", "原因"),
    "reason": ("reason", "原因"),
    "issue_type": ("issue_type", "问题类型"),
    "specific_problem": ("specific_problem", "具体诉求", "问题"),
    "urgency": ("urgency", "紧急"),
    "need_human_review": ("need_human_review", "人工"),
}


def check_must_ask_for(reply: str, fields: list[str]) -> list[str]:
    missing = []
    for field in fields:
        aliases = MUST_ASK_FIELD_ALIASES.get(field, (field,))
        if not any(alias.lower() in (reply or "").lower() for alias in aliases):
            missing.append(field)
    return missing


def check_must_not_reveal(reply: str, terms: list[str]) -> list[str]:
    return [t for t in terms if t and t.lower() in (reply or "").lower()]
