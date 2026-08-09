import threading

from app.services.cost_recorder import (
    current_cost_intent,
    get_cost_recorder,
    set_cost_intent,
)


def test_record_accumulates_and_flush_clears():
    recorder = get_cost_recorder()
    recorder.reset_for_test()
    recorder.record("gpt-4o", "order_query", 100, 50, 0.01)
    recorder.record("gpt-4o", "order_query", 200, 100, 0.02)
    recorder.record("deepseek-chat", "general", 50, 10, 0.001)
    batch = recorder.flush()
    assert len(batch) == 2
    by_key = {item["model"]: item for item in batch}
    assert by_key["gpt-4o"]["call_count"] == 2
    assert by_key["gpt-4o"]["input_tokens"] == 300
    assert by_key["gpt-4o"]["output_tokens"] == 150
    assert by_key["gpt-4o"]["total_tokens"] == 450
    assert abs(by_key["gpt-4o"]["estimated_cost"] - 0.03) < 1e-6
    assert recorder.flush() == []


def test_record_is_thread_safe():
    recorder = get_cost_recorder()
    recorder.reset_for_test()

    def worker():
        for _ in range(200):
            recorder.record("gpt-4o", "order_query", 1, 1, 0.001)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    batch = recorder.flush()
    assert batch[0]["call_count"] == 800
    assert batch[0]["input_tokens"] == 800


def test_cost_intent_context_defaults_to_general():
    assert current_cost_intent() == "general"
    set_cost_intent("refund_request")
    assert current_cost_intent() == "refund_request"
    set_cost_intent("general")
    assert current_cost_intent() == "general"
