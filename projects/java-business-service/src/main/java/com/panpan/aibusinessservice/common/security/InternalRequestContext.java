package com.panpan.aibusinessservice.common.security;

public record InternalRequestContext(
        String traceId,
        String caller,
        String userId,
        String tenantId
) {
}
