package com.panpan.aibusinessservice.common.rate;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;

public interface ToolRateLimiter {
    void check(InternalRequestContext context, String method, String uri);
}
