import pytest

from app.services.cost_recorder import get_cost_recorder
from app.services.cost_sync import PendingCostSync


class _FakeClient:
    def __init__(self) -> None:
        self.synced: list[list[dict]] = []
        self.fail_next = False

    def sync_cost_records(self, records: list[dict]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("java down")
        self.synced.append(records)


def _record_batch() -> None:
    recorder = get_cost_recorder()
    recorder.reset_for_test()
    recorder.record("gpt-4o", "order_query", 100, 50, 0.01)


def test_sync_once_flushes_batch() -> None:
    client = _FakeClient()
    sync = PendingCostSync(client=client, interval_seconds=30.0)
    _record_batch()
    assert sync.sync_once() == 1
    assert len(client.synced) == 1
    assert client.synced[0][0]["model"] == "gpt-4o"
    # 成功后再 flush 应为空
    assert get_cost_recorder().flush() == []


def test_sync_once_restores_batch_on_failure() -> None:
    client = _FakeClient()
    client.fail_next = True
    sync = PendingCostSync(client=client, interval_seconds=30.0)
    _record_batch()
    assert sync.sync_once() == 0
    # 失败后桶被回填，下次可重试
    assert sync.sync_once() == 1
    assert len(client.synced) == 1


def test_sync_once_noop_when_empty() -> None:
    client = _FakeClient()
    sync = PendingCostSync(client=client, interval_seconds=30.0)
    get_cost_recorder().reset_for_test()
    assert sync.sync_once() == 0
    assert client.synced == []
