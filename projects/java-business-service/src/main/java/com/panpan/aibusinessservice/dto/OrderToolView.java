package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.Order;
import java.math.BigDecimal;
import java.time.LocalDateTime;

public record OrderToolView(
        String orderId,
        String orderStatus,
        String paymentStatus,
        String logisticsMessage,
        String latestEvent,
        boolean canCreateTicket,
        BigDecimal refundAmount,
        LocalDateTime refundedAt,
        String refundReason,
        LocalDateTime canceledAt,
        String cancelReason,
        String userVisibleSummary
) {
    public static OrderToolView from(Order order) {
        return new OrderToolView(
                order.getOrderId(),
                order.getOrderStatus(),
                order.getPaymentStatus(),
                order.getLogisticsMessage(),
                order.getLatestEvent(),
                order.isCanCreateTicket(),
                order.getRefundAmount(),
                order.getRefundedAt(),
                order.getRefundReason(),
                order.getCanceledAt(),
                order.getCancelReason(),
                buildSummary(order)
        );
    }

    private static String buildSummary(Order order) {
        return order.getLogisticsMessage();
    }
}
