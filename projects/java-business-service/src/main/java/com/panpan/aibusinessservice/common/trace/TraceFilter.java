package com.panpan.aibusinessservice.common.trace;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.UUID;
import java.util.regex.Pattern;
import org.slf4j.MDC;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class TraceFilter extends OncePerRequestFilter {
    private static final Logger log = LoggerFactory.getLogger(TraceFilter.class);
    private static final Pattern TRACE_ID_PATTERN = Pattern.compile("^[A-Za-z0-9._:-]{8,128}$");
    private static final String TRACE_ID_MDC_KEY = "trace_id";
    private static final String TRACE_ID_ATTRIBUTE = TraceFilter.class.getName() + ".TRACE_ID";

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        String traceId = resolveTraceId(request.getHeader(TraceHeaders.TRACE_ID));
        request.setAttribute(TRACE_ID_ATTRIBUTE, traceId);
        response.setHeader(TraceHeaders.TRACE_ID, traceId);
        MDC.put(TRACE_ID_MDC_KEY, traceId);
        long startNanos = System.nanoTime();
        boolean failed = false;
        log.info("java_request_started trace_id={} method={} path={}", traceId, request.getMethod(), request.getRequestURI());
        try {
            filterChain.doFilter(request, response);
        } catch (ServletException | IOException | RuntimeException exception) {
            failed = true;
            log.warn(
                    "java_request_failed trace_id={} method={} path={} elapsed_ms={}",
                    traceId,
                    request.getMethod(),
                    request.getRequestURI(),
                    elapsedMillis(startNanos),
                    exception
            );
            throw exception;
        } finally {
            if (!failed) {
                log.info(
                        "java_request_finished trace_id={} method={} path={} status_code={} elapsed_ms={}",
                        traceId,
                        request.getMethod(),
                        request.getRequestURI(),
                        response.getStatus(),
                        elapsedMillis(startNanos)
                );
            }
            MDC.remove(TRACE_ID_MDC_KEY);
        }
    }

    public static String currentTraceId(HttpServletRequest request) {
        Object traceId = request.getAttribute(TRACE_ID_ATTRIBUTE);
        if (traceId instanceof String value && !value.isBlank()) {
            return value;
        }
        String headerValue = request.getHeader(TraceHeaders.TRACE_ID);
        if (headerValue == null || headerValue.isBlank()) {
            return "-";
        }
        return headerValue.trim();
    }

    private String resolveTraceId(String incomingTraceId) {
        if (incomingTraceId != null) {
            String trimmedTraceId = incomingTraceId.trim();
            if (TRACE_ID_PATTERN.matcher(trimmedTraceId).matches()) {
                return trimmedTraceId;
            }
        }
        return UUID.randomUUID().toString().replace("-", "");
    }

    private static double elapsedMillis(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0;
    }
}
