package com.panpan.aibusinessservice.common.trace;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Pattern;

public record JavaBusinessTracingPlan(
        String traceId,
        JavaBusinessFlow flow,
        JavaBusinessSpanSpec rootSpan,
        List<JavaBusinessSpanSpec> spans,
        List<JavaBusinessEventSpec> events,
        List<JavaBusinessMetricSpec> metrics
) {
    private static final String SERVICE_NAME = "java-business-service";
    private static final Pattern ATTRIBUTE_KEY_PATTERN = Pattern.compile("^[A-Za-z][A-Za-z0-9_.-]*$");
    private static final Pattern ATTRIBUTE_KEY_UNSAFE_PATTERN = Pattern.compile("[^A-Za-z0-9_.-]+");
    private static final String DEFAULT_TRACE_ID = "-";

    private static final List<String> PROTECTED_ATTRIBUTE_KEYS = List.of(
            "service.name",
            "app.flow",
            "app.trace_id",
            "http.route",
            "http.method",
            "upstream.python_span"
    );
    private static final List<String> SENSITIVE_ATTRIBUTE_KEYS = List.of(
            "api_key",
            "authorization",
            "cookie",
            "set_cookie",
            "password",
            "internal_token",
            "idempotency_key",
            "request_body",
            "raw_payload",
            "ticket_description",
            "title",
            "order_payload",
            "ticket_payload",
            "user_id",
            "customer_id",
            "receiver_phone"
    );
    private static final List<String> HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS = List.of(
            "trace_id",
            "span_id",
            "parent_span_id",
            "app.trace_id",
            "request_id",
            "user_id",
            "customer_id",
            "order_id",
            "ticket_id",
            "idempotency_key"
    );

    public enum JavaBusinessFlow {
        QUERY_ORDER("query_order", "GET", "/internal/orders/{orderId}", "java.orders.get"),
        CREATE_TICKET("create_ticket", "POST", "/internal/tickets", "java.tickets.create");

        private final String value;
        private final String method;
        private final String route;
        private final String upstreamPythonSpan;

        JavaBusinessFlow(String value, String method, String route, String upstreamPythonSpan) {
            this.value = value;
            this.method = method;
            this.route = route;
            this.upstreamPythonSpan = upstreamPythonSpan;
        }

        public String value() {
            return value;
        }

        public String method() {
            return method;
        }

        public String route() {
            return route;
        }

        public String upstreamPythonSpan() {
            return upstreamPythonSpan;
        }
    }

    public enum JavaBusinessSpanKind {
        SERVER,
        INTERNAL,
        CLIENT
    }

    public enum JavaBusinessEventSeverity {
        INFO,
        WARNING,
        ERROR
    }

    public enum JavaBusinessMetricKind {
        COUNTER,
        HISTOGRAM
    }

    public record JavaBusinessSpanSpec(
            String name,
            JavaBusinessSpanKind kind,
            String parentName,
            Map<String, Object> attributes
    ) {
    }

    public record JavaBusinessEventSpec(
            String name,
            String spanName,
            JavaBusinessEventSeverity severity,
            Map<String, Object> attributes
    ) {
    }

    public record JavaBusinessMetricSpec(
            String name,
            JavaBusinessMetricKind kind,
            String unit,
            Map<String, Object> attributes,
            String description
    ) {
    }

    public static JavaBusinessTracingPlan queryOrder(String traceId) {
        return build(JavaBusinessFlow.QUERY_ORDER, traceId);
    }

    public static JavaBusinessTracingPlan createTicket(String traceId) {
        return build(JavaBusinessFlow.CREATE_TICKET, traceId);
    }

    public List<String> spanNames() {
        return spans.stream().map(JavaBusinessSpanSpec::name).toList();
    }

    public List<String> eventNames() {
        return events.stream().map(JavaBusinessEventSpec::name).toList();
    }

    public List<String> metricNames() {
        return metrics.stream().map(JavaBusinessMetricSpec::name).toList();
    }

    public static Map<String, Object> safeSpanAttributes(
            JavaBusinessFlow flow,
            String traceId,
            Map<String, Object> extraAttributes
    ) {
        Map<String, Object> attributes = baseAttributes(flow, traceId);
        if (extraAttributes != null) {
            mergeExtraAttributes(attributes, extraAttributes);
        }
        return Map.copyOf(attributes);
    }

    public static Map<String, Object> safeMetricAttributes(Map<String, Object> attributes) {
        Map<String, Object> metricAttributes = new LinkedHashMap<>();
        if (attributes == null) {
            return Map.of();
        }
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            String key = normalizeAttributeKey(entry.getKey());
            if (key == null
                    || SENSITIVE_ATTRIBUTE_KEYS.contains(key)
                    || HIGH_CARDINALITY_METRIC_ATTRIBUTE_KEYS.contains(key)) {
                continue;
            }
            safeAttributeValue(entry.getValue()).ifPresent(value -> metricAttributes.put(key, value));
        }
        return Map.copyOf(metricAttributes);
    }

    private static JavaBusinessTracingPlan build(JavaBusinessFlow flow, String traceId) {
        String selectedTraceId = normalizeTraceId(traceId);
        JavaBusinessSpanSpec rootSpan = span(
                "java.http.request",
                JavaBusinessSpanKind.SERVER,
                null,
                flow,
                selectedTraceId
        );
        List<JavaBusinessSpanSpec> spans = new ArrayList<>();
        spans.add(rootSpan);
        spans.add(span("java.internal.auth.resolve", JavaBusinessSpanKind.INTERNAL, rootSpan.name(), flow, selectedTraceId));
        spans.add(span("java.rate_limit.check", JavaBusinessSpanKind.INTERNAL, rootSpan.name(), flow, selectedTraceId));

        if (flow == JavaBusinessFlow.QUERY_ORDER) {
            spans.add(span("java.order.controller.get", JavaBusinessSpanKind.INTERNAL, rootSpan.name(), flow, selectedTraceId));
            spans.add(span("java.order.service.query", JavaBusinessSpanKind.INTERNAL, "java.order.controller.get", flow, selectedTraceId));
            spans.add(span("java.redis.order_cache.get", JavaBusinessSpanKind.CLIENT, "java.order.service.query", flow, selectedTraceId));
            spans.add(span("java.mybatis.orders.select", JavaBusinessSpanKind.CLIENT, "java.order.service.query", flow, selectedTraceId));
            spans.add(span("java.order.permission.check", JavaBusinessSpanKind.INTERNAL, "java.order.service.query", flow, selectedTraceId));
        } else {
            spans.add(span("java.ticket.controller.create", JavaBusinessSpanKind.INTERNAL, rootSpan.name(), flow, selectedTraceId));
            spans.add(span("java.ticket.request.validation", JavaBusinessSpanKind.INTERNAL, "java.ticket.controller.create", flow, selectedTraceId));
            spans.add(span("java.ticket.service.create", JavaBusinessSpanKind.INTERNAL, "java.ticket.controller.create", flow, selectedTraceId));
            spans.add(span("java.order.permission.check", JavaBusinessSpanKind.INTERNAL, "java.ticket.service.create", flow, selectedTraceId));
            spans.add(span("java.redis.ticket_idempotency.get", JavaBusinessSpanKind.CLIENT, "java.ticket.service.create", flow, selectedTraceId));
            spans.add(span("java.mybatis.ticket.select_by_idempotency", JavaBusinessSpanKind.CLIENT, "java.ticket.service.create", flow, selectedTraceId));
            spans.add(span("java.mybatis.ticket.insert", JavaBusinessSpanKind.CLIENT, "java.ticket.service.create", flow, selectedTraceId));
            spans.add(span("java.mybatis.ticket_event.insert", JavaBusinessSpanKind.CLIENT, "java.ticket.service.create", flow, selectedTraceId));
            spans.add(span("java.redis.ticket_idempotency.set", JavaBusinessSpanKind.CLIENT, "java.ticket.service.create", flow, selectedTraceId));
        }

        return new JavaBusinessTracingPlan(
                selectedTraceId,
                flow,
                rootSpan,
                List.copyOf(spans),
                events(flow),
                metrics(flow)
        );
    }

    private static JavaBusinessSpanSpec span(
            String name,
            JavaBusinessSpanKind kind,
            String parentName,
            JavaBusinessFlow flow,
            String traceId
    ) {
        Map<String, Object> attributes = safeSpanAttributes(
                flow,
                traceId,
                Map.of("span.name", name)
        );
        return new JavaBusinessSpanSpec(name, kind, parentName, attributes);
    }

    private static List<JavaBusinessEventSpec> events(JavaBusinessFlow flow) {
        List<JavaBusinessEventSpec> events = new ArrayList<>();
        events.add(event("internal_auth_failed", "java.internal.auth.resolve", JavaBusinessEventSeverity.ERROR));
        events.add(event("tool_rate_limited", "java.rate_limit.check", JavaBusinessEventSeverity.WARNING));
        events.add(event("redis_unavailable", "java.rate_limit.check", JavaBusinessEventSeverity.WARNING));
        if (flow == JavaBusinessFlow.QUERY_ORDER) {
            events.add(event("order_cache_miss", "java.redis.order_cache.get", JavaBusinessEventSeverity.INFO));
            events.add(event("order_not_found", "java.mybatis.orders.select", JavaBusinessEventSeverity.WARNING));
            events.add(event("order_access_denied", "java.order.permission.check", JavaBusinessEventSeverity.WARNING));
        } else {
            events.add(event("validation_failed", "java.ticket.request.validation", JavaBusinessEventSeverity.ERROR));
            events.add(event("order_access_denied", "java.order.permission.check", JavaBusinessEventSeverity.WARNING));
            events.add(event("ticket_idempotency_replayed", "java.redis.ticket_idempotency.get", JavaBusinessEventSeverity.INFO));
            events.add(event("idempotency_key_conflict", "java.ticket.service.create", JavaBusinessEventSeverity.WARNING));
            events.add(event("transaction_rolled_back", "java.ticket.service.create", JavaBusinessEventSeverity.ERROR));
        }
        return List.copyOf(events);
    }

    private static JavaBusinessEventSpec event(
            String name,
            String spanName,
            JavaBusinessEventSeverity severity
    ) {
        return new JavaBusinessEventSpec(
                name,
                spanName,
                severity,
                Map.of("event.name", name, "event.severity", severity.name())
        );
    }

    private static List<JavaBusinessMetricSpec> metrics(JavaBusinessFlow flow) {
        Map<String, Object> attributes = safeMetricAttributes(baseAttributes(flow, DEFAULT_TRACE_ID));
        List<JavaBusinessMetricSpec> metrics = new ArrayList<>();
        metrics.add(metric("java_business.request.count", JavaBusinessMetricKind.COUNTER, "1", attributes, "Number of Java business internal API requests."));
        metrics.add(metric("java_business.request.duration", JavaBusinessMetricKind.HISTOGRAM, "ms", attributes, "Distribution of Java business internal API latency."));
        metrics.add(metric("java_business.db.query.duration", JavaBusinessMetricKind.HISTOGRAM, "ms", attributes, "Distribution of MyBatis database query latency."));
        metrics.add(metric("java_business.redis.operation.duration", JavaBusinessMetricKind.HISTOGRAM, "ms", attributes, "Distribution of Redis cache/rate/idempotency operation latency."));
        if (flow == JavaBusinessFlow.QUERY_ORDER) {
            metrics.add(metric("java_business.order.query.count", JavaBusinessMetricKind.COUNTER, "1", attributes, "Number of order query operations."));
        } else {
            metrics.add(metric("java_business.ticket.created.count", JavaBusinessMetricKind.COUNTER, "1", attributes, "Number of created tickets."));
            metrics.add(metric("java_business.idempotency.replay.count", JavaBusinessMetricKind.COUNTER, "1", attributes, "Number of idempotency replayed ticket creations."));
        }
        return List.copyOf(metrics);
    }

    private static JavaBusinessMetricSpec metric(
            String name,
            JavaBusinessMetricKind kind,
            String unit,
            Map<String, Object> attributes,
            String description
    ) {
        return new JavaBusinessMetricSpec(name, kind, unit, attributes, description);
    }

    private static Map<String, Object> baseAttributes(JavaBusinessFlow flow, String traceId) {
        Map<String, Object> attributes = new LinkedHashMap<>();
        attributes.put("service.name", SERVICE_NAME);
        attributes.put("app.flow", flow.value());
        attributes.put("app.trace_id", normalizeTraceId(traceId));
        attributes.put("http.method", flow.method());
        attributes.put("http.route", flow.route());
        attributes.put("upstream.python_span", flow.upstreamPythonSpan());
        return attributes;
    }

    private static void mergeExtraAttributes(Map<String, Object> attributes, Map<String, Object> extraAttributes) {
        for (Map.Entry<String, Object> entry : extraAttributes.entrySet()) {
            String key = normalizeAttributeKey(entry.getKey());
            if (key == null
                    || PROTECTED_ATTRIBUTE_KEYS.contains(key)
                    || SENSITIVE_ATTRIBUTE_KEYS.contains(key)
                    || attributes.containsKey(key)) {
                continue;
            }
            safeAttributeValue(entry.getValue()).ifPresent(value -> attributes.put(key, value));
        }
    }

    private static String normalizeTraceId(String traceId) {
        if (traceId == null || traceId.isBlank()) {
            return DEFAULT_TRACE_ID;
        }
        return traceId.trim();
    }

    private static String normalizeAttributeKey(String key) {
        if (key == null || key.isBlank()) {
            return null;
        }
        String normalized = ATTRIBUTE_KEY_UNSAFE_PATTERN
                .matcher(key.trim().replace(" ", "_"))
                .replaceAll("_")
                .replaceAll("^[_.-]+|[_.-]+$", "")
                .toLowerCase(Locale.ROOT);
        if (normalized.isBlank() || !ATTRIBUTE_KEY_PATTERN.matcher(normalized).matches()) {
            return null;
        }
        return normalized;
    }

    private static java.util.Optional<Object> safeAttributeValue(Object value) {
        if (value instanceof String text) {
            String trimmed = text.trim();
            return trimmed.isBlank() ? java.util.Optional.empty() : java.util.Optional.of(trimmed);
        }
        if (value instanceof Integer || value instanceof Long || value instanceof Boolean) {
            return java.util.Optional.of(value);
        }
        if (value instanceof Float floatValue && Float.isFinite(floatValue)) {
            return java.util.Optional.of(floatValue);
        }
        if (value instanceof Double doubleValue && Double.isFinite(doubleValue)) {
            return java.util.Optional.of(doubleValue);
        }
        return java.util.Optional.empty();
    }
}
