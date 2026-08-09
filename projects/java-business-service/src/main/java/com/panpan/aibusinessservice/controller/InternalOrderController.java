package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.service.OrderService;
import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.dto.OrderToolView;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import jakarta.servlet.http.HttpServletRequest;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/orders")
public class InternalOrderController {
    private final InternalRequestResolver requestResolver;
    private final OrderService orderService;

    public InternalOrderController(
            InternalRequestResolver requestResolver,
            OrderService orderService
    ) {
        this.requestResolver = requestResolver;
        this.orderService = orderService;
    }

    @GetMapping("/{orderId}")
    public ApiResponse<OrderToolView> getOrder(
            @PathVariable String orderId,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(orderService.queryOrder(orderId, context), context.traceId());
    }

    @PostMapping("/{orderId}/refund")
    public ApiResponse<OrderToolView> refundOrder(
            @PathVariable String orderId,
            @RequestBody(required = false) Map<String, String> body,
            @RequestHeader(value = TraceHeaders.IDEMPOTENCY_KEY, required = false) String idempotencyKey,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        if (body == null) {
            throw new BusinessException(BusinessErrorCode.REFUND_REASON_REQUIRED);
        }
        String reason = body.get("reason");
        if (reason == null || reason.isBlank()) {
            throw new BusinessException(BusinessErrorCode.REFUND_REASON_REQUIRED);
        }
        return ApiResponse.ok(
                orderService.refundOrder(orderId, reason, context, idempotencyKey),
                context.traceId()
        );
    }

    @PostMapping("/{orderId}/cancel")
    public ApiResponse<OrderToolView> cancelOrder(
            @PathVariable String orderId,
            @RequestBody(required = false) Map<String, String> body,
            @RequestHeader(value = TraceHeaders.IDEMPOTENCY_KEY, required = false) String idempotencyKey,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        if (body == null) {
            throw new BusinessException(BusinessErrorCode.CANCEL_REASON_REQUIRED);
        }
        String reason = body.get("reason");
        if (reason == null || reason.isBlank()) {
            throw new BusinessException(BusinessErrorCode.CANCEL_REASON_REQUIRED);
        }
        return ApiResponse.ok(
                orderService.cancelOrder(orderId, reason, context, idempotencyKey),
                context.traceId()
        );
    }
}
