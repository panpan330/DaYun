package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;

import com.panpan.aibusinessservice.common.trace.JavaBusinessTracingPlan;
import com.panpan.aibusinessservice.common.trace.JavaBusinessTracingPlan.JavaBusinessFlow;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JavaBusinessTracingPlanTest {
    @Test
    void queryOrderPlanAlignsJavaInternalSpansWithPythonClientSpan() {
        JavaBusinessTracingPlan plan = JavaBusinessTracingPlan.queryOrder("trace-java-order-001");

        assertThat(plan.traceId()).isEqualTo("trace-java-order-001");
        assertThat(plan.flow()).isEqualTo(JavaBusinessFlow.QUERY_ORDER);
        assertThat(plan.spanNames()).containsExactly(
                "java.http.request",
                "java.internal.auth.resolve",
                "java.rate_limit.check",
                "java.order.controller.get",
                "java.order.service.query",
                "java.redis.order_cache.get",
                "java.mybatis.orders.select",
                "java.order.permission.check"
        );
        assertThat(plan.eventNames()).contains(
                "internal_auth_failed",
                "tool_rate_limited",
                "order_cache_miss",
                "order_not_found",
                "order_access_denied"
        );
        assertThat(plan.metricNames()).contains(
                "java_business.request.count",
                "java_business.request.duration",
                "java_business.db.query.duration",
                "java_business.redis.operation.duration",
                "java_business.order.query.count"
        );
        assertThat(plan.rootSpan().attributes())
                .containsEntry("service.name", "java-business-service")
                .containsEntry("app.flow", "query_order")
                .containsEntry("http.route", "/internal/orders/{orderId}")
                .containsEntry("http.method", "GET")
                .containsEntry("upstream.python_span", "java.orders.get");
    }

    @Test
    void createTicketPlanCoversValidationTransactionMyBatisRedisAndIdempotency() {
        JavaBusinessTracingPlan plan = JavaBusinessTracingPlan.createTicket("trace-java-ticket-001");

        assertThat(plan.spanNames()).containsExactly(
                "java.http.request",
                "java.internal.auth.resolve",
                "java.rate_limit.check",
                "java.ticket.controller.create",
                "java.ticket.request.validation",
                "java.ticket.service.create",
                "java.order.permission.check",
                "java.redis.ticket_idempotency.get",
                "java.mybatis.ticket.select_by_idempotency",
                "java.mybatis.ticket.insert",
                "java.mybatis.ticket_event.insert",
                "java.redis.ticket_idempotency.set"
        );
        assertThat(plan.eventNames()).contains(
                "validation_failed",
                "ticket_idempotency_replayed",
                "idempotency_key_conflict",
                "transaction_rolled_back"
        );
        assertThat(plan.metricNames()).contains(
                "java_business.ticket.created.count",
                "java_business.idempotency.replay.count"
        );
        assertThat(plan.rootSpan().attributes())
                .containsEntry("http.route", "/internal/tickets")
                .containsEntry("http.method", "POST")
                .containsEntry("upstream.python_span", "java.tickets.create");
    }

    @Test
    void safeSpanAttributesKeepTraceMetadataButOmitSecretsAndSensitivePayloads() {
        Map<String, Object> attributes = JavaBusinessTracingPlan.safeSpanAttributes(
                JavaBusinessFlow.CREATE_TICKET,
                "trace-safe-001",
                Map.of(
                        "internal_token", "local-dev-internal-token",
                        "Authorization", "Bearer secret",
                        "idempotency_key", "idem-001",
                        "ticket_description", "private complaint",
                        "user_id", "U1001",
                        "custom.retry_count", 2,
                        "custom.cache_hit", true,
                        "app.trace_id", "wrong-trace",
                        "raw_payload", Map.of("too", "large")
                )
        );

        assertThat(attributes)
                .containsEntry("app.trace_id", "trace-safe-001")
                .containsEntry("app.flow", "create_ticket")
                .containsEntry("custom.retry_count", 2)
                .containsEntry("custom.cache_hit", true);
        assertThat(attributes).doesNotContainKeys(
                "internal_token",
                "authorization",
                "idempotency_key",
                "ticket_description",
                "user_id",
                "raw_payload"
        );
    }

    @Test
    void safeMetricAttributesUseLowCardinalityFieldsOnly() {
        Map<String, Object> metricAttributes = JavaBusinessTracingPlan.safeMetricAttributes(
                Map.of(
                        "app.flow", "query_order",
                        "http.route", "/internal/orders/{orderId}",
                        "status", "ok",
                        "trace_id", "trace-001",
                        "order_id", "A1001",
                        "idempotency_key", "idem-001",
                        "user_id", "U1001",
                        "internal_token", "local-dev-internal-token"
                )
        );

        assertThat(metricAttributes)
                .containsEntry("app.flow", "query_order")
                .containsEntry("http.route", "/internal/orders/{orderId}")
                .containsEntry("status", "ok");
        assertThat(metricAttributes).doesNotContainKeys(
                "trace_id",
                "order_id",
                "idempotency_key",
                "user_id",
                "internal_token"
        );
    }
}
