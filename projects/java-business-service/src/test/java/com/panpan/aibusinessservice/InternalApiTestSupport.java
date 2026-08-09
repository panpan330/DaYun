package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

final class InternalApiTestSupport {
    static final String TRACE_ID = "trace-stage7-test";
    static final String INTERNAL_TOKEN = "local-dev-internal-token";

    private InternalApiTestSupport() {
    }

    static MockHttpServletRequestBuilder withInternalHeaders(MockHttpServletRequestBuilder request) {
        return request
                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                .header(TraceHeaders.CALLER, "ai-service")
                .header(TraceHeaders.USER_ID, "U1001")
                .header(TraceHeaders.TENANT_ID, "default")
                .header(TraceHeaders.INTERNAL_TOKEN, INTERNAL_TOKEN);
    }
}
