import pytest

from app.core.production_monitoring import (
    AlertRuleSpec,
    MonitoringCatalog,
    ProductionMetricSpec,
    build_default_monitoring_catalog,
    build_metric_catalog_summary,
    evaluate_alert_rule,
    find_high_cardinality_metric_labels,
    format_monitoring_catalog,
)


def test_default_monitoring_catalog_contains_ai_production_core_domains() -> None:
    catalog = build_default_monitoring_catalog()
    summary = build_metric_catalog_summary(catalog)

    assert summary.metric_count == 15
    assert summary.alert_rule_count == 6
    assert summary.domain_counts == {
        "cost": 2,
        "http": 2,
        "java": 1,
        "llm": 2,
        "rag": 3,
        "resilience": 2,
        "safety": 1,
        "tool": 2,
    }
    assert summary.alert_severity_counts == {"critical": 2, "warning": 4}
    assert "http.server.requests" in summary.alert_metric_names
    assert "llm.failures" in summary.alert_metric_names
    assert find_high_cardinality_metric_labels(catalog) == {}


def test_format_monitoring_catalog_is_readable_for_learning_and_review() -> None:
    catalog = build_default_monitoring_catalog()

    lines = format_monitoring_catalog(catalog)

    assert lines[0] == "Production monitoring catalog"
    assert "metrics: 15" in lines
    assert "alert_rules: 6" in lines
    assert any("llm.calls type=counter domain=llm" in line for line in lines)
    assert any("High LLM failure rate" in line for line in lines)


def test_metric_spec_rejects_high_cardinality_labels() -> None:
    with pytest.raises(ValueError, match="high-cardinality"):
        ProductionMetricSpec(
            name="http.server.requests",
            metric_type="counter",
            domain="http",
            unit="requests",
            description="Bad metric with user id label.",
            labels=["route", "user_id"],
        )


def test_monitoring_catalog_rejects_alert_for_unknown_metric() -> None:
    metric = ProductionMetricSpec(
        name="http.server.requests",
        metric_type="counter",
        domain="http",
        unit="requests",
        description="Total HTTP requests.",
        labels=["route", "method", "status_code_class"],
    )
    alert = AlertRuleSpec(
        name="Unknown metric alert",
        metric_name="missing.metric",
        severity="warning",
        comparator=">=",
        threshold=1,
        window="15m",
        description="This alert references a missing metric.",
        runbook_hint="Fix the metric name.",
    )

    with pytest.raises(ValueError, match="unknown metrics"):
        MonitoringCatalog(metrics=[metric], alert_rules=[alert])


def test_alert_rule_rejects_noisy_short_window_info_alert() -> None:
    with pytest.raises(ValueError, match="too noisy"):
        AlertRuleSpec(
            name="Noisy info alert",
            metric_name="http.server.requests",
            severity="info",
            comparator=">=",
            threshold=1,
            window="5m",
            description="Too noisy for production.",
            runbook_hint="Use dashboard instead of alert.",
        )


def test_evaluate_alert_rule_uses_comparator_and_threshold() -> None:
    rule = AlertRuleSpec(
        name="High p95 request latency",
        metric_name="http.server.duration",
        severity="warning",
        comparator=">=",
        threshold=5000,
        window="15m",
        description="HTTP p95 latency is above 5 seconds.",
        runbook_hint="Check latency breakdown.",
    )

    assert evaluate_alert_rule(rule, current_value=5000) is True
    assert evaluate_alert_rule(rule, current_value=4999.99) is False
