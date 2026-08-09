from datetime import timedelta
from pathlib import Path

import json

import pytest

from app.agents.eval_suite import AgentEvalRunReport, AgentEvalSuiteReport
from app.evaluation.eval_platform import (
    EvalDatasetRegistry,
    EvalRunContext,
    compare_eval_run_snapshots,
    find_eval_dataset_manifest,
    format_eval_dataset_registry,
    format_eval_regression_report,
    build_agent_eval_run_snapshot,
    load_eval_dataset_registry,
)
from app.evaluation.snapshot_store import SnapshotStore


REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "evaluation" / "datasets.json"
)


def test_load_eval_dataset_registry_reads_current_local_manifests() -> None:
    registry = load_eval_dataset_registry(REGISTRY_PATH)

    assert registry.registry_version == "stage10-v1"
    assert [dataset.name for dataset in registry.datasets] == [
        "agent_eval",
        "rag_retrieval_eval",
        "rag_answer_eval",
    ]
    agent_dataset = find_eval_dataset_manifest(
        registry,
        name="agent_eval",
        version="stage6-v1",
    )
    assert agent_dataset.cases_path == "data/agent_eval/agent_cases.json"
    assert agent_dataset.frozen is True


def test_format_eval_dataset_registry_is_readable() -> None:
    registry = load_eval_dataset_registry(REGISTRY_PATH)

    lines = format_eval_dataset_registry(registry)

    assert lines[0] == "Evaluation dataset registry"
    assert "registry_version: stage10-v1" in lines
    assert any("agent_eval:stage6-v1" in line for line in lines)
    assert any("baseline=agent_eval_stage6_v1_baseline" in line for line in lines)


def test_eval_dataset_registry_rejects_duplicate_name_and_version() -> None:
    with pytest.raises(ValueError, match="duplicate name/version"):
        EvalDatasetRegistry.model_validate(
            {
                "registry_version": "test",
                "datasets": [
                    {
                        "name": "agent_eval",
                        "version": "v1",
                        "task_type": "agent",
                        "cases_path": "a.json",
                    },
                    {
                        "name": "agent_eval",
                        "version": "v1",
                        "task_type": "agent",
                        "cases_path": "b.json",
                    },
                ],
            }
        )


def test_build_agent_eval_run_snapshot_turns_suite_report_into_platform_snapshot() -> None:
    report = _agent_report(
        passed=True,
        intent_failed=0,
        route_failed=0,
    )
    context = _context(run_id="candidate-001", candidate_version="prompt-v2")

    snapshot = build_agent_eval_run_snapshot(report, context=context)

    assert snapshot.context.run_id == "candidate-001"
    assert snapshot.evaluated_check_count == 20
    assert snapshot.passed_check_count == 20
    assert snapshot.failed_check_count == 0
    assert snapshot.passed is True
    assert snapshot.metric_map()["suite_pass_rate"].value == 1.0
    assert snapshot.metric_map()["check_pass_rate"].value == 1.0
    assert snapshot.metric_map()["failed_checks"].direction == "lower_is_better"


def test_compare_eval_run_snapshots_reports_no_regression_when_candidate_is_equal() -> None:
    baseline = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="baseline-001", candidate_version="prompt-v1"),
    )
    candidate = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="candidate-001", candidate_version="prompt-v2"),
    )

    regression = compare_eval_run_snapshots(baseline, candidate)

    assert regression.regressed is False
    assert regression.blocking_reasons == []
    assert all(not metric.regressed for metric in regression.metric_comparisons)


def test_compare_eval_run_snapshots_reports_metric_and_status_regression() -> None:
    baseline = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="baseline-001", candidate_version="prompt-v1"),
    )
    candidate = build_agent_eval_run_snapshot(
        _agent_report(passed=False, intent_failed=2, route_failed=0),
        context=_context(run_id="candidate-001", candidate_version="prompt-v2"),
    )

    regression = compare_eval_run_snapshots(baseline, candidate)
    lines = format_eval_regression_report(regression)

    assert regression.regressed is True
    assert "metric_regressed:check_pass_rate" in regression.blocking_reasons
    assert "metric_regressed:failed_checks" in regression.blocking_reasons
    assert "overall_status_regressed" in regression.blocking_reasons
    assert "failed_check_count_increased" in regression.blocking_reasons
    assert "regressed: true" in lines


def test_compare_eval_run_snapshots_rejects_different_dataset_versions() -> None:
    baseline = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(
            run_id="baseline-001",
            candidate_version="prompt-v1",
            dataset_version="stage6-v1",
        ),
    )
    candidate = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(
            run_id="candidate-001",
            candidate_version="prompt-v2",
            dataset_version="stage6-v2",
        ),
    )

    with pytest.raises(ValueError, match="different dataset versions"):
        compare_eval_run_snapshots(baseline, candidate)


def test_snapshot_store_saves_and_loads_latest(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.json")
    first = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="run-001", candidate_version="prompt-v1"),
    )
    second = build_agent_eval_run_snapshot(
        _agent_report(passed=False, intent_failed=2, route_failed=0),
        context=_context(run_id="run-002", candidate_version="prompt-v2"),
    )

    store.save(first)
    store.save(second)

    latest = store.load_latest()
    assert latest is not None
    assert latest.context.run_id == "run-002"
    assert latest.context.candidate_version == "prompt-v2"
    assert latest.passed is False
    assert latest.failed_check_count == 2
    assert latest.metric_map()["failed_checks"].value == 2.0


def test_snapshot_store_load_latest_returns_none_without_existing_file(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "missing.json")

    assert store.load_latest() is None


def test_snapshot_store_save_is_atomic_and_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "snapshots.json"
    store = SnapshotStore(path)
    first = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="run-001", candidate_version="prompt-v1"),
    )
    store.save(first)
    original_content = path.read_text(encoding="utf-8")

    real_write_text = Path.write_text

    def failing_write_text(self, *args, **kwargs):  # noqa: ANN001
        if self.suffix == ".tmp":
            raise OSError("simulated write failure")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    second = build_agent_eval_run_snapshot(
        _agent_report(passed=True, intent_failed=0, route_failed=0),
        context=_context(run_id="run-002", candidate_version="prompt-v2"),
    )
    with pytest.raises(OSError, match="simulated write failure"):
        store.save(second)
    monkeypatch.undo()

    assert path.read_text(encoding="utf-8") == original_content
    assert store.load_latest() is not None
    assert store.load_latest().context.run_id == "run-001"


def test_snapshot_store_keeps_only_latest_max_snapshots(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.json", max_snapshots=2)
    for index in range(4):
        snapshot = build_agent_eval_run_snapshot(
            _agent_report(passed=True, intent_failed=0, route_failed=0),
            context=_context(run_id=f"run-{index:03d}", candidate_version=f"prompt-v{index}"),
        )
        store.save(snapshot)

    raw_items = json.loads(store.path.read_text(encoding="utf-8"))
    assert len(raw_items) == 2
    assert raw_items[-1]["context"]["run_id"] == "run-003"
    assert raw_items[0]["context"]["run_id"] == "run-002"
    assert store.load_latest().context.run_id == "run-003"
    store.save(
        build_agent_eval_run_snapshot(
            _agent_report(passed=True, intent_failed=0, route_failed=0),
            context=_context(run_id="run-004", candidate_version="prompt-v4"),
        )
    )
    raw_items = json.loads(store.path.read_text(encoding="utf-8"))
    assert len(raw_items) == 2
    assert raw_items[-1]["context"]["run_id"] == "run-004"
    assert store.load_latest().context.run_id == "run-004"


def _context(
    *,
    run_id: str,
    candidate_version: str,
    dataset_version: str = "stage6-v1",
) -> EvalRunContext:
    return EvalRunContext(
        run_id=run_id,
        dataset_name="agent_eval",
        dataset_version=dataset_version,
        candidate_version=candidate_version,
        model_name="fake",
        prompt_version=candidate_version,
        code_version="local-test",
    )


def _agent_report(
    *,
    passed: bool,
    intent_failed: int,
    route_failed: int,
) -> AgentEvalRunReport:
    suite_reports = [
        _suite_report(
            name="intent",
            case_count=10,
            failed_case_count=intent_failed,
        ),
        _suite_report(
            name="route",
            case_count=10,
            failed_case_count=route_failed,
        ),
    ]
    failed_suite_count = sum(1 for suite in suite_reports if not suite.passed)
    return AgentEvalRunReport(
        cases_path="data/agent_eval/agent_cases.json",
        selected_case_count=10,
        case_filter="all",
        suite_count=2,
        passed_suite_count=2 - failed_suite_count,
        failed_suite_count=failed_suite_count,
        passed=passed,
        suite_reports=suite_reports,
    )


def _suite_report(
    *,
    name: str,
    case_count: int,
    failed_case_count: int,
) -> AgentEvalSuiteReport:
    return AgentEvalSuiteReport(
        name=name,
        title=f"{name} suite",
        case_count=case_count,
        failed_case_count=failed_case_count,
        passed=failed_case_count == 0,
        summary_lines=[],
        bad_case_lines=[],
    )


def test_eval_run_context_carries_started_at() -> None:
    report = _agent_report(passed=True, intent_failed=0, route_failed=0)
    context = _context(run_id="run-started-at", candidate_version="prompt-v1")
    assert context.started_at is None

    snapshot = build_agent_eval_run_snapshot(report, context=context)

    assert snapshot.context.started_at is not None
    assert snapshot.context.started_at.utcoffset() == timedelta(0)


def test_snapshot_store_load_all_returns_ordered_snapshots(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.json")
    for index in range(3):
        store.save(
            build_agent_eval_run_snapshot(
                _agent_report(passed=True, intent_failed=0, route_failed=0),
                context=_context(
                    run_id=f"run-{index:03d}",
                    candidate_version=f"prompt-v{index}",
                ),
            )
        )

    loaded = store.load_all()

    assert [item.context.run_id for item in loaded] == ["run-000", "run-001", "run-002"]
    assert all(
        loaded[index].context.started_at <= loaded[index + 1].context.started_at
        for index in range(len(loaded) - 1)
    )
