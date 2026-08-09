from app.services.conversation_history import build_history_window


def _rounds(count: int) -> list[dict]:
    """Alternating user/assistant transcript, oldest first."""
    messages = []
    for i in range(count):
        messages.append({"role": "user", "content": f"u{i}", "created_at": f"2026-08-08T09:{i:02d}:00Z"})
        messages.append({"role": "assistant", "content": f"a{i}", "created_at": f"2026-08-08T09:{i:02d}:30Z"})
    return messages


def test_keeps_all_rounds_within_limit_in_order():
    result = build_history_window(_rounds(5), max_rounds=10, max_tokens=10000)

    assert result == ["u0", "a0", "u1", "a1", "u2", "a2", "u3", "a3", "u4"]


def test_drops_oldest_rounds_beyond_max_rounds():
    result = build_history_window(_rounds(12), max_rounds=10, max_tokens=100000)

    # 20 条中取最近 10 轮 = 20 条，但末尾 assistant 去掉 → 19 条（u2..a10 + u11）
    expected = []
    for i in range(2, 12):
        expected.extend([f"u{i}", f"a{i}"])
    expected.pop()  # 末尾 a11 被去掉
    assert result == expected


def test_drops_oldest_until_within_token_budget():
    # 每条 content 长 200 字符，中文估算 100 token；预算 300 → 只能留 3 条
    long = "长" * 200
    messages = []
    for i in range(4):
        messages.append({"role": "user", "content": long, "created_at": f"2026-08-08T09:{i:02d}:00Z"})
        messages.append({"role": "assistant", "content": long, "created_at": f"2026-08-08T09:{i:02d}:30Z"})

    result = build_history_window(messages, max_rounds=10, max_tokens=300)

    assert len(result) == 3  # 300 token / 100 per item
    assert result[0].startswith("长")


def test_drops_trailing_assistant_so_window_ends_with_user():
    result = build_history_window(_rounds(2), max_rounds=10, max_tokens=10000)

    assert result == ["u0", "a0", "u1"]  # 末尾 assistant a1 被去掉


def test_empty_input_returns_empty():
    assert build_history_window([], max_rounds=10, max_tokens=3000) == []


def test_single_user_message_passthrough():
    messages = [{"role": "user", "content": "你好", "created_at": "2026-08-08T09:00:00Z"}]

    result = build_history_window(messages, max_rounds=10, max_tokens=3000)

    assert result == ["你好"]
