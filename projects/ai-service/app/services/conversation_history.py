"""Window assembly for the conversation history fed to the model.

The input is the stored transcript (oldest first, user/assistant alternating).
The output is a list of plain message texts (newest last, ending with a user
message) bounded by two budgets:
  - max_rounds: at most N user/assistant pairs
  - max_tokens: cumulative estimated token budget, dropping oldest first
"""


def build_history_window(
    messages: list[dict],
    *,
    max_rounds: int = 10,
    max_tokens: int = 3000,
) -> list[str]:
    """Return up to max_rounds of recent transcript texts within max_tokens.

    ``messages`` items use ``{"role": "user"|"assistant", "content": str}``
    (created_at is optional and ignored here; ordering is caller-provided).
    """
    if not messages:
        return []

    # 1. Rounds window: keep the last 2*max_rounds messages (user+assistant pairs).
    tail = messages[-2 * max_rounds :]

    # 2. End with a user message: a trailing assistant turn is half of a round
    #    and would read as an unanswered model turn.
    while tail and tail[-1].get("role") == "assistant":
        tail = tail[:-1]
    if not tail:
        return []

    # 3. Token budget: drop oldest until under budget (estimated tokens).
    texts = [str(message.get("content", "")) for message in tail]
    estimated = [_estimate_tokens(text) for text in texts]
    while sum(estimated) > max_tokens and len(texts) > 1:
        texts = texts[1:]
        estimated = [_estimate_tokens(text) for text in texts]
    return texts


def _estimate_tokens(text: str) -> int:
    # Chinese-heavy transcripts: ~2 chars per token. ASCII mixes in too; a
    # simple char/2 estimate is good enough for a window budget.
    return max(1, len(text) // 2)
