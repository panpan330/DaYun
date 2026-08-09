package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.OrderListItemView;
import com.panpan.aibusinessservice.dto.OrderToolView;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.service.AuthService;
import com.panpan.aibusinessservice.service.OrderQueryService;
import com.panpan.aibusinessservice.service.OrderService;
import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/orders")
public class OrderController {
    private final AuthService authService;
    private final OrderQueryService orderQueryService;
    private final OrderService orderService;

    public OrderController(AuthService authService, OrderQueryService orderQueryService, OrderService orderService) {
        this.authService = authService;
        this.orderQueryService = orderQueryService;
        this.orderService = orderService;
    }

    @GetMapping
    public ApiResponse<List<OrderListItemView>> listOrders(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        return ApiResponse.ok(
                orderQueryService.listVisibleOrders(currentUser),
                TraceFilter.currentTraceId(servletRequest)
        );
    }

    @PostMapping("/{orderId}/refund")
    public ApiResponse<OrderToolView> refundOrder(
            @PathVariable String orderId,
            @RequestBody(required = false) Map<String, String> body,
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        if (body == null) {
            throw new BusinessException(BusinessErrorCode.REFUND_REASON_REQUIRED);
        }
        String reason = body.get("reason");
        if (reason == null || reason.isBlank()) {
            throw new BusinessException(BusinessErrorCode.REFUND_REASON_REQUIRED);
        }
        String traceId = TraceFilter.currentTraceId(servletRequest);
        InternalRequestContext context = new InternalRequestContext(
                traceId,
                "api",
                currentUser.userId(),
                currentUser.tenantId()
        );
        return ApiResponse.ok(
                orderService.refundOrder(orderId, reason, context, null),
                traceId
        );
    }

    @PostMapping("/{orderId}/cancel")
    public ApiResponse<OrderToolView> cancelOrder(
            @PathVariable String orderId,
            @RequestBody(required = false) Map<String, String> body,
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        if (body == null) {
            throw new BusinessException(BusinessErrorCode.CANCEL_REASON_REQUIRED);
        }
        String reason = body.get("reason");
        if (reason == null || reason.isBlank()) {
            throw new BusinessException(BusinessErrorCode.CANCEL_REASON_REQUIRED);
        }
        String traceId = TraceFilter.currentTraceId(servletRequest);
        InternalRequestContext context = new InternalRequestContext(
                traceId,
                "api",
                currentUser.userId(),
                currentUser.tenantId()
        );
        return ApiResponse.ok(
                orderService.cancelOrder(orderId, reason, context, null),
                traceId
        );
    }
}
