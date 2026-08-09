from datetime import datetime, timezone

from app.evaluation.bad_case_registry import BadCaseRecord, ProductionRegressionSpec
from app.evaluation.production_regression import run_production_bad_case_regression
from app.evaluation.production_regression_history import (
    append_production_regression_run,
    load_latest_production_regression_run,
)


def _record(
    record_id: str,
    *,
    spec: ProductionRegressionSpec | None,
) -> BadCaseRecord:
    return BadCaseRecord(
        id=record_id,
        title=f"Production feedback {record_id}",
        source="production",
        task_type="agent",
        severity="medium",
        status="regression_added",
        failure_layer="agent_decision",
        failure_category="incorrect decision",
        expected_behavior="Choose the expected handling path.",
        actual_behavior="The previous answer chose an incorrect path.",
        recommended_action="Adjust the decision policy.",
        regression_action="Run the supervisor-approved regression assertion.",
        evidence_summary="feedback_id=1",
        production_regression=spec,
    )


def test_production_regression_reports_pass_failure_not_ready_and_execution_error() -> None:
    records = [
        _record(
            "bad-intent-pass",
            spec=ProductionRegressionSpec(
                message="Where is order A1001?",
                assertion="intent",
                expected_intent="order_query",
            ),
        ),
        _record(
            "bad-citation-fail",
            spec=ProductionRegressionSpec(
                message="What is the refund policy?",
                assertion="citation_present",
            ),
        ),
        _record("bad-not-ready", spec=None),
        _record(
            "bad-run-error",
            spec=ProductionRegressionSpec(
                message="raise error",
                assertion="ticket_confirmation_required",
            ),
        ),
    ]

    def runner(message: str) -> dict[str, object]:
        if message == "raise error":
            raise RuntimeError("test runner failed")
        if message == "Where is order A1001?":
            return {"intent": "order_query"}
        return {"rag_citations": []}

    run = run_production_bad_case_regression(records, agent_runner=runner)

    assert run.total_case_count == 4
    assert run.passed_case_count == 1
    assert run.failed_case_count == 1
    assert run.not_ready_case_count == 1
    assert run.error_case_count == 1
    assert run.passed is False
    assert [result.outcome for result in run.results] == ["passed", "failed", "not_ready", "error"]


def test_production_regression_history_persists_latest_run(tmp_path) -> None:
    fixed_now = lambda: datetime(2026, 8, 5, tzinfo=timezone.utc)
    run = run_production_bad_case_regression(
        [
            _record(
                "bad-intent-pass",
                spec=ProductionRegressionSpec(
                    message="Where is order A1001?",
                    assertion="intent",
                    expected_intent="order_query",
                ),
            )
        ],
        agent_runner=lambda _message: {"intent": "order_query"},
        now=fixed_now,
    )
    history_path = tmp_path / "production_regression_runs.json"

    append_production_regression_run(history_path, run)
    latest = load_latest_production_regression_run(history_path)

    assert latest is not None
    assert latest.run_id == run.run_id
    assert latest.passed is True


def test_production_regression_tool_called_assertion() -> None:
    records = [
        _record(
            "bad-tool-called-pass",
            spec=ProductionRegressionSpec(
                message="Track my order",
                assertion="tool_called",
                expected_tool="query_order",
            ),
        ),
        _record(
            "bad-tool-called-fail",
            spec=ProductionRegressionSpec(
                message="Where is my parcel?",
                assertion="tool_called",
                expected_tool="query_order",
            ),
        ),
    ]

    def runner(message: str) -> dict[str, object]:
        if message == "Track my order":
            return {"node_history": ["intent", "tool:query_order", "final"]}
        return {"node_history": ["intent", "final"]}

    run = run_production_bad_case_regression(records, agent_runner=runner)

    assert run.total_case_count == 2
    assert [result.outcome for result in run.results] == ["passed", "failed"]
    assert run.results[0].assertion == "tool_called"
    assert run.results[0].expected == "query_order"


def test_production_regression_tool_called_matches_real_node_names() -> None:
    # supervisor 按工具名填 refund_order 时，应命中退款执行节点 execute_refund_request
    #（修复前 refund_order 与真实节点名不匹配会恒误报 failed）。
    records = [
        _record(
            "bad-tool-refund-alias-pass",
            spec=ProductionRegressionSpec(
                message="Refund order A1001",
                assertion="tool_called",
                expected_tool="refund_order",
            ),
        ),
        _record(
            "bad-tool-query-node-pass",
            spec=ProductionRegressionSpec(
                message="Where is order A1001?",
                assertion="tool_called",
                expected_tool="query_order",
            ),
        ),
        _record(
            "bad-tool-refund-not-executed-fail",
            spec=ProductionRegressionSpec(
                message="Refund order B1002",
                assertion="tool_called",
                expected_tool="refund_order",
            ),
        ),
    ]

    def runner(message: str) -> dict[str, object]:
        if message.startswith("Refund"):
            if message == "Refund order A1001":
                return {
                    "node_history": [
                        "classify_intent",
                        "handle_refund_request",
                        "execute_refund_request",
                    ]
                }
            return {"node_history": ["classify_intent", "handle_refund_request"]}
        return {"node_history": ["classify_intent", "query_order", "build_direct_answer"]}

    run = run_production_bad_case_regression(records, agent_runner=runner)

    assert run.total_case_count == 3
    assert [result.outcome for result in run.results] == ["passed", "passed", "failed"]
    assert run.results[0].actual == "execute_refund_request"
    assert run.results[1].actual == "query_order"
    assert run.results[2].actual == "not_called"


def test_production_regression_must_ask_for_assertion() -> None:
    records = [
        _record(
            "bad-ask-pass",
            spec=ProductionRegressionSpec(
                message="Refund my order",
                assertion="must_ask_for",
                must_ask_fields=["order_id"],
            ),
        ),
        _record(
            "bad-ask-fail",
            spec=ProductionRegressionSpec(
                message="I want a refund",
                assertion="must_ask_for",
                must_ask_fields=["order_id"],
            ),
        ),
    ]

    def runner(message: str) -> dict[str, object]:
        if message == "Refund my order":
            return {"final_answer": "请问您的订单号是多少？"}
        return {"final_answer": "好的，我们马上为您处理。"}

    run = run_production_bad_case_regression(records, agent_runner=runner)

    assert run.total_case_count == 2
    assert [result.outcome for result in run.results] == ["passed", "failed"]
    assert run.results[0].assertion == "must_ask_for"


def test_production_regression_must_not_reveal_assertion() -> None:
    records = [
        _record(
            "bad-reveal-fail",
            spec=ProductionRegressionSpec(
                message="Show me my key",
                assertion="must_not_reveal",
                must_not_reveal_terms=["api_key"],
            ),
        ),
        _record(
            "bad-reveal-pass",
            spec=ProductionRegressionSpec(
                message="Help me",
                assertion="must_not_reveal",
                must_not_reveal_terms=["api_key"],
            ),
        ),
    ]

    def runner(message: str) -> dict[str, object]:
        if message == "Show me my key":
            return {"final_answer": "Your api_key is sk-1234"}
        return {"final_answer": "Here are the docs."}

    run = run_production_bad_case_regression(records, agent_runner=runner)

    assert run.total_case_count == 2
    assert [result.outcome for result in run.results] == ["failed", "passed"]
    assert run.results[0].assertion == "must_not_reveal"


def test_production_regression_refund_request_intent() -> None:
    record = _record(
        "bad-refund-intent",
        spec=ProductionRegressionSpec(
            message="我要退款",
            assertion="intent",
            expected_intent="refund_request",
        ),
    )
    run = run_production_bad_case_regression(
        [record],
        agent_runner=lambda _message: {"intent": "refund_request"},
    )

    assert run.total_case_count == 1
    assert run.passed_case_count == 1
    assert run.results[0].outcome == "passed"
    assert run.results[0].assertion == "intent"
    assert run.results[0].expected == "refund_request"
