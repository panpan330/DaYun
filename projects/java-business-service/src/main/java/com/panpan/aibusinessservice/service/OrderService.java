package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.OrderToolView;

public interface OrderService {
    OrderToolView queryOrder(String orderId, InternalRequestContext context);

    OrderToolView refundOrder(
            String orderId,
            String reason,
            InternalRequestContext context,
            String idempotencyKey
    );

    OrderToolView cancelOrder(
            String orderId,
            String reason,
            InternalRequestContext context,
            String idempotencyKey
    );
}
