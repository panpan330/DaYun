package com.panpan.aibusinessservice.common.security;

import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.common.rate.ToolRateLimiter;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.config.InternalApiProperties;
import jakarta.servlet.http.HttpServletRequest;
import java.util.regex.Pattern;
import org.springframework.stereotype.Component;

@Component
public class InternalRequestResolver {
    private static final Pattern TRACE_ID_PATTERN = Pattern.compile("^[A-Za-z0-9._:-]{8,128}$");
    private static final Pattern CALLER_PATTERN = Pattern.compile("^[a-z][a-z0-9-]{1,63}$");
    private static final Pattern IDENTITY_PATTERN = Pattern.compile("^[A-Za-z0-9._:-]{1,64}$");

    private final InternalApiProperties properties;
    private final ToolRateLimiter toolRateLimiter;

    public InternalRequestResolver(InternalApiProperties properties, ToolRateLimiter toolRateLimiter) {
        this.properties = properties;
        this.toolRateLimiter = toolRateLimiter;
    }

    public InternalRequestContext resolve(HttpServletRequest request) {
        String traceId = requiredHeader(request, TraceHeaders.TRACE_ID, TRACE_ID_PATTERN);
        String caller = requiredHeader(request, TraceHeaders.CALLER, CALLER_PATTERN);
        String userId = requiredHeader(request, TraceHeaders.USER_ID, IDENTITY_PATTERN);
        String tenantId = requiredHeader(request, TraceHeaders.TENANT_ID, IDENTITY_PATTERN);
        String token = requiredHeader(request, TraceHeaders.INTERNAL_TOKEN);

        if (!properties.allowedCaller().equals(caller)
                || properties.token() == null
                || !properties.token().equals(token)) {
            throw new BusinessException(BusinessErrorCode.INTERNAL_AUTH_FAILED);
        }

        InternalRequestContext context = new InternalRequestContext(traceId, caller, userId, tenantId);
        toolRateLimiter.check(context, request.getMethod(), request.getRequestURI());
        return context;
    }

    private String requiredHeader(HttpServletRequest request, String name) {
        String value = request.getHeader(name);
        if (value == null || value.isBlank()) {
            throw new BusinessException(BusinessErrorCode.INTERNAL_AUTH_FAILED);
        }
        return value.trim();
    }

    private String requiredHeader(HttpServletRequest request, String name, Pattern pattern) {
        String value = requiredHeader(request, name);
        if (!pattern.matcher(value).matches()) {
            throw new BusinessException(BusinessErrorCode.INTERNAL_AUTH_FAILED);
        }
        return value;
    }
}
