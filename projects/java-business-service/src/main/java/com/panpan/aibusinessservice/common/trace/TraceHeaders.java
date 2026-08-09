package com.panpan.aibusinessservice.common.trace;

public final class TraceHeaders {
    public static final String TRACE_ID = "X-Trace-Id";
    public static final String CALLER = "X-Caller";
    public static final String USER_ID = "X-User-Id";
    public static final String TENANT_ID = "X-Tenant-Id";
    public static final String INTERNAL_TOKEN = "X-Internal-Token";
    public static final String IDEMPOTENCY_KEY = "Idempotency-Key";

    private TraceHeaders() {
    }
}
