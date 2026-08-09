import json
from datetime import datetime, timezone

import httpx
import pytest

from app.evaluation.rag_retrieval_history import RagRetrievalRun
from app.services.java_rag_eval_client import JavaRagEvalClient, JavaRagEvalClientError


def _client(handler):
    transport = httpx.MockTransport(handler)
    return JavaRagEvalClient(base_url="http://java.test", timeout_seconds=5, transport=transport)


def _sample_run():
    now = datetime.now(timezone.utc)
    return RagRetrievalRun(
        run_id="rag-retrieval-abc",
        retriever_mode="keyword",
        started_at=now,
        completed_at=now,
        top_k=5,
        hit_rate=0.8,
        recall=0.75,
        precision=0.7,
        mrr=0.6,
        case_count=12,
        results=[{"id": "c1", "hit": True}],
    )


def test_save_run_posts_to_internal_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t1"})

    client = _client(handler)
    client.save_run(_sample_run())

    assert captured["url"].endswith("/internal/rag-eval-runs")
    assert captured["json"]["run_id"] == "rag-retrieval-abc"
    assert captured["json"]["retriever"] == "keyword"
    assert captured["json"]["top_k"] == 5
    assert captured["json"]["hit_rate"] == 0.8
    assert captured["json"]["details_json"] == '[{"id": "c1", "hit": true}]'


def test_save_run_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"success": False, "code": "INTERNAL", "message": "boom", "data": None, "trace_id": "t1"})

    with pytest.raises(JavaRagEvalClientError) as exc_info:
        _client(handler).save_run(_sample_run())
    assert exc_info.value.status_code == 500


def test_save_run_raises_on_request_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(JavaRagEvalClientError) as exc_info:
        _client(handler).save_run(_sample_run())
    assert exc_info.value.status_code == 0


def test_list_runs_returns_data():
    def handler(request):
        return httpx.Response(200, json={"success": True, "code": "OK", "message": "ok", "data": [{"run_id": "run-1"}], "trace_id": "t1"})

    runs = _client(handler).list_runs(limit=10)
    assert runs == [{"run_id": "run-1"}]


def test_latest_run_returns_data_or_none():
    def handler(request):
        return httpx.Response(200, json={"success": True, "code": "OK", "message": "ok", "data": {"run_id": "run-2"}, "trace_id": "t1"})

    run = _client(handler).latest_run("vector")
    assert run == {"run_id": "run-2"}


def test_cli_call_always_sends_trace_id_header():
    """CLI callers (no request trace context) must still send X-Trace-Id,
    otherwise Java internal auth rejects with 401."""
    from app.core.trace import reset_trace_id, set_trace_id

    token = set_trace_id("-")
    try:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["trace_id"] = request.headers.get("X-Trace-Id")
            return httpx.Response(200, json={"success": True, "code": "OK", "message": "ok", "data": None, "trace_id": "t1"})

        _client(handler).save_run(_sample_run())
    finally:
        reset_trace_id(token)

    trace_id = captured["trace_id"]
    assert isinstance(trace_id, str)
    assert len(trace_id) >= 8
    assert trace_id.startswith("sync-")
