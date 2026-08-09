package com.panpan.aibusinessservice.exception;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.ConstraintViolationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {
    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ApiResponse<Object>> handleBusinessException(
            BusinessException exception,
            HttpServletRequest request
    ) {
        BusinessErrorCode code = exception.errorCode();
        return ResponseEntity
                .status(code.status())
                .body(ApiResponse.error(code.name(), exception.getMessage(), traceId(request)));
    }

    @ExceptionHandler({MethodArgumentNotValidException.class, ConstraintViolationException.class})
    public ResponseEntity<ApiResponse<Object>> handleValidationException(
            Exception exception,
            HttpServletRequest request
    ) {
        log.warn("Request validation failed, trace_id={}, reason={}", traceId(request), exception.getMessage());
        return ResponseEntity
                .status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(ApiResponse.error(
                        BusinessErrorCode.TICKET_REQUEST_INVALID.name(),
                        BusinessErrorCode.TICKET_REQUEST_INVALID.defaultMessage(),
                        traceId(request)
                ));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiResponse<Object>> handleUnexpectedException(
            Exception exception,
            HttpServletRequest request
    ) {
        log.error("Unexpected Java business service error, trace_id={}", traceId(request), exception);
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error("JAVA_SERVICE_ERROR", "Java 业务服务内部错误。", traceId(request)));
    }

    private String traceId(HttpServletRequest request) {
        return TraceFilter.currentTraceId(request);
    }
}
