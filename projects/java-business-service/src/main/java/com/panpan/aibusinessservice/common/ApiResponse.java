package com.panpan.aibusinessservice.common;

import com.fasterxml.jackson.annotation.JsonProperty;

public record ApiResponse<T>(
        boolean success,
        String code,
        String message,
        T data,
        @JsonProperty("trace_id") String traceId
) {
    public static <T> ApiResponse<T> ok(T data, String traceId) {
        return new ApiResponse<>(true, "OK", "OK", data, traceId);
    }

    public static ApiResponse<Object> error(String code, String message, String traceId) {
        return new ApiResponse<>(false, code, message, null, traceId);
    }
}
