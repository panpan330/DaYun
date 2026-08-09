from pathlib import Path

import logging
from datetime import datetime, timezone

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.trace import TRACE_ID_HEADER
from app.routers import evaluation as evaluation_router
from app.routers.evaluation import get_evaluation_actor, get_evaluation_registry_path
from app.routers.evaluation import get_bad_case_registry_path, get_production_regression_history_path
from app.routers.evaluation import get_eval_snapshot_store_path
from app.routers.evaluation import get_rag_retrieval_history_path
from app.routers.evaluation import get_agent_cases_path
from app.agents.intent_evaluation import AgentEvalDataset
from app.evaluation.bad_case_registry import BadCaseRegistry
from app.services.java_feedback_client import JavaFeedbackContext
from app.routers.evaluation import _load_or_generate_bad_cases
from app.evaluation.eval_platform import (
    EvalDatasetManifest,
    EvalMetric,
    EvalRunContext,
    EvalRunSnapshot,
)
from app.evaluation.snapshot_store import SnapshotStore
from app.agents.eval_suite import run_agent_eval_suites
from app.evaluation.production_regression import ProductionRegressionRun
from app.evaluation.production_regression import run_production_bad_case_regression
from app.evaluation.production_regression_history import append_production_regression_run
from app.services.console_agent_service import ConsoleAgentActor


def test_evaluation_overview_returns_local_registry_run_and_bad_cases(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_eval_snapshot_store_path] = lambda: tmp_path / "snapshots.json"
    response = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-eval-overview"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["registry_version"] == "stage10-v1"
    assert len(data["datasets"]) == 3
    assert data["latest_run"]["run_id"] == "local-agent-eval-latest"
    assert data["latest_run"]["dataset_name"] == "agent_eval"
    assert data["latest_run"]["selected_case_count"] > 0
    assert len(data["latest_run"]["suites"]) == 6
    assert data["bad_case_summary"]["record_count"] == len(data["bad_cases"])
    assert data["trace_id"] == "trace-eval-overview"


def test_evaluation_overview_reports_missing_registry(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_evaluation_registry_path] = lambda: tmp_path / "missing.json"

    response = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-eval-missing"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "EVALUATION_DATA_NOT_FOUND",
        "message": "本地评估数据文件不存在，无法生成评估看板。",
        "trace_id": "trace-eval-missing",
    }


def test_production_bad_cases_merge_with_generated_local_evaluation_cases(tmp_path: Path) -> None:
    registry_path = tmp_path / "bad_cases.json"
    registry_path.write_text(
        """
        {
          "schema_version": "stage10.bad_case_registry.v1",
          "records": [{
            "id": "bad_production_feedback_stage11_feedback_1_agent_decision",
            "title": "Production feedback",
            "source": "production",
            "task_type": "agent",
            "severity": "medium",
            "status": "regression_added",
            "source_case_id": "feedback_1",
            "failure_layer": "agent_decision",
            "failure_category": "decision",
            "expected_behavior": "Escalate correctly.",
            "actual_behavior": "Did not escalate.",
            "recommended_action": "Adjust policy.",
            "regression_action": "Add a test.",
            "evidence_summary": "feedback_id=1"
          }]
        }
        """,
        encoding="utf-8",
    )
    cases_path = Path(__file__).parents[1] / "data" / "agent_eval" / "agent_cases.json"
    report = run_agent_eval_suites(cases_path)
    dataset = EvalDatasetManifest(
        name="agent_eval",
        version="stage6-v1",
        task_type="agent",
        description="test",
        frozen=True,
        cases_path="data/agent_eval/agent_cases.json",
    )

    records, generated = _load_or_generate_bad_cases(registry_path, run_report=report, dataset=dataset)

    assert generated is True
    assert any(record.source == "production" for record in records)


def test_supervisor_can_run_and_persist_formal_production_regression(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "bad_cases.json"
    registry_path.write_text(
        """
        {
          "schema_version": "stage10.bad_case_registry.v1",
          "records": [{
            "id": "bad_production_feedback_stage11_feedback_1_agent_decision",
            "title": "Production feedback",
            "source": "production",
            "task_type": "agent",
            "severity": "medium",
            "status": "regression_added",
            "source_case_id": "feedback_1",
            "failure_layer": "agent_decision",
            "failure_category": "decision",
            "expected_behavior": "Route order questions correctly.",
            "actual_behavior": "The route was incorrect.",
            "recommended_action": "Adjust the decision policy.",
            "regression_action": "Run an intent regression.",
            "evidence_summary": "feedback_id=1",
            "production_regression": {
              "message": "Where is order A1001?",
              "assertion": "intent",
              "expected_intent": "order_query"
            }
          }]
        }
        """,
        encoding="utf-8",
    )
    history_path = tmp_path / "production_regression_runs.json"
    app.dependency_overrides[get_evaluation_actor] = lambda: ConsoleAgentActor(
        user_id="SUP1001", tenant_id="default", roles=("supervisor",)
    )
    app.dependency_overrides[get_bad_case_registry_path] = lambda: registry_path
    app.dependency_overrides[get_production_regression_history_path] = lambda: history_path
    monkeypatch.setattr(
        evaluation_router,
        "run_production_bad_case_regression",
        lambda records: run_production_bad_case_regression(
            records,
            agent_runner=lambda _message: {"intent": "order_query"},
        ),
    )

    response = client.post(
        "/api/ai/evaluation/runs/production-regression",
        headers={TRACE_ID_HEADER: "trace-production-regression"},
    )

    assert response.status_code == 200
    assert response.json()["passed"] is True
    assert response.json()["results"][0]["outcome"] == "passed"
    assert history_path.exists()


def _overview_snapshot(context: EvalRunContext, *, passed: bool) -> EvalRunSnapshot:
    failed_checks = 0 if passed else 2
    passed_checks = 20 - failed_checks
    return EvalRunSnapshot(
        context=context,
        evaluated_check_count=20,
        passed_check_count=passed_checks,
        failed_check_count=failed_checks,
        passed=passed,
        metrics=[
            EvalMetric(name="suite_pass_rate", value=1.0 if passed else 0.5),
            EvalMetric(name="check_pass_rate", value=1.0 if passed else 0.9),
            EvalMetric(
                name="failed_suites",
                value=0.0 if passed else 1.0,
                direction="lower_is_better",
            ),
            EvalMetric(
                name="failed_checks",
                value=float(failed_checks),
                direction="lower_is_better",
            ),
        ],
    )


def _override_snapshot_store_and_builder(
    app: FastAPI,
    tmp_path: Path,
    monkeypatch,
    state: dict[str, bool],
) -> None:
    app.dependency_overrides[get_eval_snapshot_store_path] = (
        lambda: tmp_path / "agent_eval_snapshots.json"
    )
    monkeypatch.setattr(
        evaluation_router,
        "build_agent_eval_run_snapshot",
        lambda report, *, context: _overview_snapshot(context, passed=state["passed"]),
    )


def test_overview_includes_baseline_comparison(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = {"passed": True}
    _override_snapshot_store_and_builder(app, tmp_path, monkeypatch, state)

    first = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-baseline-first"},
    )
    assert first.status_code == 200
    assert first.json()["baseline_comparison"] is None

    state["passed"] = False
    second = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-baseline-second"},
    )
    assert second.status_code == 200
    comparison = second.json()["baseline_comparison"]
    assert comparison is not None
    assert comparison["dataset_name"] == "agent_eval"
    assert comparison["dataset_version"] == "stage6-v1"
    assert comparison["baseline_run_id"] == "local-agent-eval-latest"
    assert comparison["candidate_run_id"] == "local-agent-eval-latest"
    assert comparison["baseline_candidate_version"] == "local-working-tree"
    assert comparison["candidate_version"] == "local-working-tree"
    assert comparison["regressed"] is True
    assert comparison["blocking_reasons"]
    assert comparison["metric_comparisons"]


def test_overview_marks_regressed_checks(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = {"passed": True}
    _override_snapshot_store_and_builder(app, tmp_path, monkeypatch, state)

    first = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-regressed-first"},
    )
    assert first.status_code == 200
    assert first.json()["baseline_comparison"] is None

    state["passed"] = False
    second = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-regressed-second"},
    )
    assert second.status_code == 200
    comparison = second.json()["baseline_comparison"]
    assert comparison is not None
    assert comparison["regressed"] is True
    regressed_metrics = [item for item in comparison["metric_comparisons"] if item["regressed"]]
    assert any(item["name"] == "check_pass_rate" for item in regressed_metrics)
    assert any(item["name"] == "failed_checks" for item in regressed_metrics)
    assert "overall_status_regressed" in comparison["blocking_reasons"]


def test_overview_skips_comparison_when_stored_baseline_dataset_version_mismatches(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "agent_eval_snapshots.json"
    app.dependency_overrides[get_eval_snapshot_store_path] = lambda: snapshot_path
    state = {"passed": True}
    _override_snapshot_store_and_builder(app, tmp_path, monkeypatch, state)
    # 预置一条旧 dataset 版本的快照，模拟 manifest 升级后遗留的历史数据。
    SnapshotStore(snapshot_path).save(
        _overview_snapshot(
            EvalRunContext(
                run_id="old-run",
                dataset_name="agent_eval",
                dataset_version="stage6-v0",
                candidate_version="old-working-tree",
            ),
            passed=True,
        )
    )

    response = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-version-mismatch"},
    )

    assert response.status_code == 200
    assert response.json()["baseline_comparison"] is None


def test_overview_degrades_gracefully_when_snapshot_save_fails(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"passed": True}
    _override_snapshot_store_and_builder(app, tmp_path, monkeypatch, state)

    def failing_save(self, snapshot):  # noqa: ANN001
        raise OSError("simulated save failure")

    monkeypatch.setattr(evaluation_router.SnapshotStore, "save", failing_save)
    caplog.set_level(logging.WARNING)

    response = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-save-failure"},
    )

    assert response.status_code == 200
    assert response.json()["baseline_comparison"] is None
    assert response.json()["latest_run"]["run_id"] == "local-agent-eval-latest"
    assert "snapshot_save_failed" in caplog.text


def test_overview_degrades_gracefully_when_snapshot_load_fails(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"passed": True}
    _override_snapshot_store_and_builder(app, tmp_path, monkeypatch, state)
    # 历史快照文件内容损坏：load_latest 抛错时应降级，看板仍 200。
    snapshot_path = tmp_path / "agent_eval_snapshots.json"
    snapshot_path.write_text('{"broken": "json"', encoding="utf-8")
    caplog.set_level(logging.WARNING)

    response = client.get(
        "/api/ai/evaluation/overview",
        headers={TRACE_ID_HEADER: "trace-load-failure"},
    )

    assert response.status_code == 200
    assert response.json()["baseline_comparison"] is None
    assert response.json()["latest_run"]["run_id"] == "local-agent-eval-latest"
    assert "snapshot_load_failed" in caplog.text


PROMOTE_INTENT_PAYLOAD = {
    "failure_layer": "intent",
    "severity": "high",
    "failure_category": "退款请求被误判为 unsupported",
    "expected_behavior": "退款请求必须路由到退款处理，不能拒绝。",
    "recommended_action": "修正意图分类器的退款识别规则。",
    "regression_action": "新增 intent 回归用例。",
    "review_note": "confirmed by supervisor",
    "regression_message": "我要申请退款，订单 A1001",
    "regression_assertion": "intent",
    "regression_expected_intent": "refund_request",
}


def _write_bad_case_registry_fixture(path: Path) -> None:
    path.write_text(
        '{"schema_version": "stage10.bad_case_registry.v1", "records": []}',
        encoding="utf-8",
    )


def _write_agent_cases_fixture(path: Path) -> None:
    path.write_text(
        '{"schema_version": "stage6.agent_eval.v1", "description": "test", "cases": []}',
        encoding="utf-8",
    )


def _intent_feedback_context() -> JavaFeedbackContext:
    return JavaFeedbackContext(
        feedback_id=1,
        conversation_id="conversation-refund-1",
        trace_id="trace-refund-1",
        reason="misclassified refund request as unsupported",
        agent_route="build_unsupported_answer",
        citation_count=0,
        human_handoff_suggested=False,
        user_message_excerpt="我要申请退款，订单 A1001",
        assistant_answer_excerpt="I am sorry, I cannot help with refunds.",
        citation_summary_json=None,
        review_status="pending",
        bad_case_id=None,
        review_note=None,
    )


def _fake_feedback_client_class(context: JavaFeedbackContext):
    class FakeJavaFeedbackClient:
        def __init__(self, *, context: JavaFeedbackContext) -> None:
            self._context = context

        @classmethod
        def from_settings(cls, settings):  # noqa: ANN001
            return cls(context=context)

        def get_context(self, feedback_id: int) -> JavaFeedbackContext:
            return self._context

        def mark_promoted(self, feedback_id: int, *, bad_case_id: str) -> JavaFeedbackContext:
            return self._context

    return FakeJavaFeedbackClient


def _promote_test_app(
    app: FastAPI,
    tmp_path: Path,
    monkeypatch,
    *,
    registry_path: Path | None = None,
    cases_path: Path | None = None,
    context: JavaFeedbackContext | None = None,
    registry_content: str | None = None,
) -> tuple[Path, Path]:
    registry_path = registry_path or (tmp_path / "bad_cases.json")
    cases_path = cases_path or (tmp_path / "agent_cases.json")
    if registry_content is None:
        _write_bad_case_registry_fixture(registry_path)
    else:
        registry_path.write_text(registry_content, encoding="utf-8")
    _write_agent_cases_fixture(cases_path)
    app.dependency_overrides[get_evaluation_actor] = lambda: ConsoleAgentActor(
        user_id="SUP1001", tenant_id="default", roles=("supervisor",)
    )
    app.dependency_overrides[get_bad_case_registry_path] = lambda: registry_path
    app.dependency_overrides[get_agent_cases_path] = lambda: cases_path
    monkeypatch.setattr(
        evaluation_router,
        "JavaFeedbackClient",
        _fake_feedback_client_class(context or _intent_feedback_context()),
    )
    return registry_path, cases_path


def test_promote_writes_bad_case_to_agent_cases(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases_path = _promote_test_app(app, tmp_path, monkeypatch)

    response = client.post(
        "/api/ai/evaluation/feedback-candidates/1/promote",
        json=PROMOTE_INTENT_PAYLOAD,
        headers={TRACE_ID_HEADER: "trace-promote-1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["written_case_id"]

    written = AgentEvalDataset.model_validate_json(cases_path.read_text(encoding="utf-8"))
    assert [case.id for case in written.cases] == [data["written_case_id"]]
    case = written.cases[0]
    assert case.inputs.message == "我要申请退款，订单 A1001"
    assert case.expected.intent == "refund_request"
    assert case.expected.intent_route == "handle_refund_request"
    assert case.metadata.task_type == "agent"
    assert case.metadata.business_domain == "production"
    assert case.metadata.case_type == "production_regression"
    assert case.metadata.difficulty == "hard"
    assert case.metadata.priority == "p0"
    assert "from_bad_case" in case.metadata.tags
    assert f"source_bad_case_id:{data['bad_case']['id']}" in case.metadata.tags


def test_promote_is_idempotent_for_same_bad_case(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, cases_path = _promote_test_app(app, tmp_path, monkeypatch)

    first = client.post(
        "/api/ai/evaluation/feedback-candidates/1/promote",
        json=PROMOTE_INTENT_PAYLOAD,
        headers={TRACE_ID_HEADER: "trace-promote-dup-1"},
    )
    assert first.status_code == 200
    assert first.json()["written_case_id"]

    second = client.post(
        "/api/ai/evaluation/feedback-candidates/1/promote",
        json=PROMOTE_INTENT_PAYLOAD,
        headers={TRACE_ID_HEADER: "trace-promote-dup-2"},
    )
    assert second.status_code == 200
    assert second.json()["written_case_id"] is None

    written = AgentEvalDataset.model_validate_json(cases_path.read_text(encoding="utf-8"))
    assert len(written.cases) == 1


REGRESSION_ADDED_REGISTRY = """
{
  "schema_version": "stage10.bad_case_registry.v1",
  "records": [{
    "id": "bad_production_feedback_stage11_feedback_1_intent",
    "title": "Production feedback 1: misroute",
    "source": "production",
    "task_type": "agent",
    "severity": "high",
    "status": "regression_added",
    "source_case_id": "feedback_1",
    "failure_layer": "intent",
    "failure_category": "misroute",
    "expected_behavior": "Refund requests must route to refund handling.",
    "actual_behavior": "The route was incorrect.",
    "recommended_action": "Adjust the intent classifier.",
    "regression_action": "Add an intent regression case.",
    "regression_dataset_name": "agent_eval",
    "regression_case_id": "feedback_1_regression_intent",
    "production_regression": {
      "message": "我要申请退款，订单 A1001",
      "assertion": "intent",
      "expected_intent": "refund_request"
    },
    "evidence_summary": "feedback_id=1"
  }]
}
"""


def _regression_added_context() -> JavaFeedbackContext:
    return JavaFeedbackContext(
        feedback_id=1,
        conversation_id="conversation-refund-1",
        trace_id="trace-refund-1",
        reason="misclassified refund request as unsupported",
        agent_route="build_unsupported_answer",
        citation_count=0,
        human_handoff_suggested=False,
        user_message_excerpt="我要申请退款，订单 A1001",
        assistant_answer_excerpt="I am sorry, I cannot help with refunds.",
        citation_summary_json=None,
        review_status="regression_added",
        bad_case_id="bad_production_feedback_stage11_feedback_1_intent",
        review_note="confirmed by supervisor",
    )


def _promote_intent_request(client: TestClient, *, trace_id: str):
    return client.post(
        "/api/ai/evaluation/feedback-candidates/1/promote",
        json=PROMOTE_INTENT_PAYLOAD,
        headers={TRACE_ID_HEADER: trace_id},
    )


def test_promote_links_regression_case_id_in_registry(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path, cases_path = _promote_test_app(app, tmp_path, monkeypatch)

    response = _promote_intent_request(client, trace_id="trace-relink-1")
    assert response.status_code == 200
    written_case_id = response.json()["written_case_id"]
    assert written_case_id

    registry = BadCaseRegistry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    )
    record = registry.records[0]
    assert record.regression_case_id == written_case_id
    assert record.regression_dataset_name == "agent_eval"
    assert record.status == "regression_added"


def test_promote_early_return_writes_back_bad_case_and_links_registry(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    # 提前返回路径：review_status=regression_added 且 registry 已有该 bad case。
    # 首次 promote 会走提前返回，此时也应补写 agent_cases 并回填 regression_case_id。
    registry_path, cases_path = _promote_test_app(
        app,
        tmp_path,
        monkeypatch,
        context=_regression_added_context(),
        registry_content=REGRESSION_ADDED_REGISTRY,
    )

    response = _promote_intent_request(client, trace_id="trace-early-1")
    assert response.status_code == 200
    written_case_id = response.json()["written_case_id"]
    assert written_case_id

    written = AgentEvalDataset.model_validate_json(cases_path.read_text(encoding="utf-8"))
    assert len(written.cases) == 1
    assert written.cases[0].expected.intent == "refund_request"
    assert written.cases[0].expected.intent_route == "handle_refund_request"

    registry = BadCaseRegistry.model_validate_json(
        registry_path.read_text(encoding="utf-8")
    )
    assert registry.records[0].regression_case_id == written_case_id

    # 再次走提前返回：写回幂等命中，不重复写，响应 written_case_id 为 None。
    second = _promote_intent_request(client, trace_id="trace-early-2")
    assert second.status_code == 200
    assert second.json()["written_case_id"] is None
    written_again = AgentEvalDataset.model_validate_json(
        cases_path.read_text(encoding="utf-8")
    )
    assert len(written_again.cases) == 1


def test_promote_degrades_when_agent_case_writeback_fails(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 写回 agent_cases.json 抛 OSError：fail-safe 降级，promote 主流程仍成功。
    _promote_test_app(app, tmp_path, monkeypatch)

    def failing_writeback(record, *, cases_path):  # noqa: ANN001
        raise OSError("simulated writeback failure")

    monkeypatch.setattr(evaluation_router, "write_bad_case_to_agent_cases", failing_writeback)
    caplog.set_level(logging.WARNING)

    response = _promote_intent_request(client, trace_id="trace-writeback-fail")
    assert response.status_code == 200
    assert response.json()["written_case_id"] is None
    assert "agent_case_writeback_failed" in caplog.text


def test_promote_degrades_when_registry_relink_fails(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 写回成功后回填 registry 失败（update_production_bad_case 抛 OSError）：
    # 降级为告警，promote 仍成功并返回已写回的 case id。
    _promote_test_app(app, tmp_path, monkeypatch)

    def failing_update(path, record):  # noqa: ANN001
        raise OSError("simulated registry update failure")

    monkeypatch.setattr(evaluation_router, "update_production_bad_case", failing_update)
    caplog.set_level(logging.WARNING)

    response = _promote_intent_request(client, trace_id="trace-relink-fail")
    assert response.status_code == 200
    assert response.json()["written_case_id"]
    assert "bad_case_relink_failed" in caplog.text


def test_history_returns_agent_and_regression_sequences(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "agent_eval_snapshots.json"
    history_path = tmp_path / "production_regression_runs.json"
    app.dependency_overrides[get_eval_snapshot_store_path] = lambda: snapshot_path
    app.dependency_overrides[get_production_regression_history_path] = lambda: history_path

    store = SnapshotStore(snapshot_path)
    store.save(
        _overview_snapshot(
            EvalRunContext(
                run_id="run-001",
                dataset_name="agent_eval",
                dataset_version="stage6-v1",
                candidate_version="prompt-v1",
                started_at=datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc),
            ),
            passed=True,
        )
    )
    store.save(
        _overview_snapshot(
            EvalRunContext(
                run_id="run-002",
                dataset_name="agent_eval",
                dataset_version="stage6-v1",
                candidate_version="prompt-v2",
                started_at=datetime(2026, 8, 7, 11, 0, 0, tzinfo=timezone.utc),
            ),
            passed=False,
        )
    )
    append_production_regression_run(
        history_path,
        ProductionRegressionRun(
            run_id="reg-001",
            started_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 8, 7, 12, 0, 5, tzinfo=timezone.utc),
            total_case_count=2,
            passed_case_count=2,
            failed_case_count=0,
            not_ready_case_count=0,
            error_case_count=0,
            passed=True,
            results=[],
        ),
    )

    response = client.get("/api/ai/evaluation/history")

    assert response.status_code == 200
    data = response.json()
    assert len(data["agent_eval"]) == 2
    assert [point["check_pass_rate"] for point in data["agent_eval"]] == [1.0, 0.9]
    assert all(point["started_at"] for point in data["agent_eval"])
    assert len(data["production_regression"]) == 1
    point = data["production_regression"][0]
    assert point["started_at"] == "2026-08-07T12:00:00+00:00"
    assert point["passed"] == 2
    assert point["total"] == 2
    assert point["pass_rate"] == 1.0


def test_history_returns_empty_arrays_when_no_data(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_eval_snapshot_store_path] = (
        lambda: tmp_path / "missing-snapshots.json"
    )
    app.dependency_overrides[get_production_regression_history_path] = (
        lambda: tmp_path / "missing-history.json"
    )
    app.dependency_overrides[get_rag_retrieval_history_path] = (
        lambda: tmp_path / "missing-rag-retrieval.json"
    )

    response = client.get("/api/ai/evaluation/history")

    assert response.status_code == 200
    assert response.json() == {"agent_eval": [], "production_regression": [], "rag_retrieval": []}


def test_history_degrades_gracefully_when_data_files_corrupted(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot_path = tmp_path / "agent_eval_snapshots.json"
    history_path = tmp_path / "production_regression_runs.json"
    app.dependency_overrides[get_eval_snapshot_store_path] = lambda: snapshot_path
    app.dependency_overrides[get_production_regression_history_path] = lambda: history_path
    app.dependency_overrides[get_rag_retrieval_history_path] = lambda: tmp_path / "missing-rag-retrieval.json"
    # 快照文件与历史文件内容损坏（截断 JSON）：history 应降级为空序列，仍 200。
    snapshot_path.write_text('{"broken": "json"', encoding="utf-8")
    history_path.write_text('{"schema_version": "stage11.production_regression_runs.v1", "runs": [{"broken"', encoding="utf-8")
    caplog.set_level(logging.WARNING)

    response = client.get("/api/ai/evaluation/history")

    assert response.status_code == 200
    assert response.json() == {"agent_eval": [], "production_regression": [], "rag_retrieval": []}
    assert "history_load_failed" in caplog.text


def test_reports_latest_agent_returns_markdown(
    app: FastAPI,
    client: TestClient,
) -> None:
    response = client.get(
        "/api/ai/evaluation/reports/latest",
        params={"type": "agent"},
        headers={TRACE_ID_HEADER: "trace-report-agent"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "agent"
    assert data["generated_at"]
    assert data["report"].startswith("# Agent Evaluation Report\n")
    assert "## Overall" in data["report"]


def test_reports_latest_regression_returns_markdown(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "production_regression_runs.json"
    app.dependency_overrides[get_production_regression_history_path] = lambda: history_path
    append_production_regression_run(
        history_path,
        ProductionRegressionRun(
            run_id="reg-001",
            started_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 8, 7, 12, 0, 5, tzinfo=timezone.utc),
            total_case_count=2,
            passed_case_count=2,
            failed_case_count=0,
            not_ready_case_count=0,
            error_case_count=0,
            passed=True,
            results=[],
        ),
    )

    response = client.get(
        "/api/ai/evaluation/reports/latest",
        params={"type": "regression"},
        headers={TRACE_ID_HEADER: "trace-report-regression"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "regression"
    assert data["report"].startswith("# Production Regression Report\n")
    assert "## Overall" in data["report"]


def test_reports_latest_not_found_when_no_data(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_production_regression_history_path] = (
        lambda: tmp_path / "missing-runs.json"
    )

    response = client.get(
        "/api/ai/evaluation/reports/latest",
        params={"type": "regression"},
        headers={TRACE_ID_HEADER: "trace-report-missing"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "REPORT_NOT_FOUND"


def test_reports_latest_rejects_unknown_type(
    app: FastAPI,
    client: TestClient,
) -> None:
    response = client.get(
        "/api/ai/evaluation/reports/latest",
        params={"type": "foo"},
        headers={TRACE_ID_HEADER: "trace-report-unknown-type"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_history_keeps_point_when_snapshot_missing_check_pass_rate(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    # 历史快照缺 check_pass_rate 指标（schema 演进遗留）：history 应保留该点、
    # check_pass_rate 为 None，不触发 KeyError/500。
    snapshot_path = tmp_path / "agent_eval_snapshots.json"
    app.dependency_overrides[get_eval_snapshot_store_path] = lambda: snapshot_path
    app.dependency_overrides[get_production_regression_history_path] = (
        lambda: tmp_path / "missing-history.json"
    )
    SnapshotStore(snapshot_path).save(
        EvalRunSnapshot(
            context=EvalRunContext(
                run_id="legacy-run",
                dataset_name="agent_eval",
                dataset_version="stage6-v0",
                candidate_version="legacy",
                started_at=datetime(2026, 8, 7, 9, 0, 0, tzinfo=timezone.utc),
            ),
            evaluated_check_count=20,
            passed_check_count=20,
            failed_check_count=0,
            passed=True,
            metrics=[EvalMetric(name="suite_pass_rate", value=1.0)],
        )
    )

    response = client.get("/api/ai/evaluation/history")

    assert response.status_code == 200
    data = response.json()
    assert len(data["agent_eval"]) == 1
    assert data["agent_eval"][0]["started_at"] == "2026-08-07T09:00:00+00:00"
    assert data["agent_eval"][0]["check_pass_rate"] is None


def test_reports_latest_agent_not_found_when_cases_missing(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
) -> None:
    # cases 文件缺失：与 regression 分支一致返回 404，而非裸 500。
    app.dependency_overrides[get_agent_cases_path] = (
        lambda: tmp_path / "missing-agent-cases.json"
    )

    response = client.get(
        "/api/ai/evaluation/reports/latest",
        params={"type": "agent"},
        headers={TRACE_ID_HEADER: "trace-report-agent-missing"},
    )

    assert response.status_code == 404
    assert response.json()["code"] == "EVALUATION_DATA_NOT_FOUND"



def test_history_includes_rag_retrieval_sequence(app: FastAPI, client: TestClient, tmp_path, monkeypatch):
    """GET /api/ai/evaluation/history 返回 rag_retrieval 序列。"""
    import json
    from datetime import datetime, timezone

    from app.evaluation.rag_retrieval_history import RagRetrievalRun, RagRetrievalRunHistory
    from app.routers.evaluation import get_rag_retrieval_history_path

    history_path = tmp_path / "rag_retrieval_runs.json"
    history = RagRetrievalRunHistory(
        runs=[
            RagRetrievalRun(
                run_id="rag-001",
                retriever_mode="keyword",
                started_at=datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc),
                completed_at=datetime(2026, 8, 7, 12, 1, tzinfo=timezone.utc),
                top_k=3,
                hit_rate=0.8,
                recall=0.7,
                precision=0.3,
                mrr=0.65,
                case_count=12,
                results=[],
            )
        ]
    )
    history_path.write_text(history.model_dump_json(indent=2) + "\n", encoding="utf-8")
    app.dependency_overrides[get_rag_retrieval_history_path] = lambda: history_path

    response = client.get("/api/ai/evaluation/history")
    assert response.status_code == 200
    data = response.json().get("data", response.json())
    rag = data.get("rag_retrieval", [])
    assert len(rag) == 1
    assert rag[0]["hit_rate"] == 0.8
    assert rag[0]["recall"] == 0.7
    assert rag[0]["mrr"] == 0.65


def test_rag_retrieval_runs_returns_java_history(
    app: FastAPI, client: TestClient, monkeypatch,
) -> None:
    from app.routers import evaluation as evaluation_module

    monkeypatch.setattr(
        evaluation_module,
        "list_rag_retrieval_runs",
        lambda limit: [{"run_id": "run-1", "retriever": "keyword"}],
    )
    response = client.get("/api/ai/evaluation/rag-retrieval-runs")
    assert response.status_code == 200
    assert response.json()["runs"][0]["run_id"] == "run-1"


def test_rag_retrieval_runs_latest_by_retriever(
    app: FastAPI, client: TestClient, monkeypatch,
) -> None:
    from app.routers import evaluation as evaluation_module

    monkeypatch.setattr(
        evaluation_module,
        "latest_rag_retrieval_run",
        lambda retriever: {"run_id": "run-2", "retriever": retriever},
    )
    response = client.get("/api/ai/evaluation/rag-retrieval-runs/latest?retriever=vector")
    assert response.status_code == 200
    assert response.json()["run"]["retriever"] == "vector"


def test_rag_retrieval_runs_degrades_gracefully(
    app: FastAPI, client: TestClient, monkeypatch,
) -> None:
    from app.services import java_rag_eval_client as client_module
    from app.services.java_rag_eval_client import JavaRagEvalClientError

    class DownClient:
        def list_runs(self, limit):
            raise JavaRagEvalClientError(status_code=0, code="RAG_EVAL_SERVICE_UNAVAILABLE", message="java down")

    monkeypatch.setattr(client_module.JavaRagEvalClient, "from_settings", lambda settings: DownClient())
    response = client.get("/api/ai/evaluation/rag-retrieval-runs")
    assert response.status_code == 200
    assert response.json()["runs"] == []
