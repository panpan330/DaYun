package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.Order;
import com.panpan.aibusinessservice.entity.OrderEvent;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface OrderMapper {
    Order selectByTenantIdAndOrderId(
            @Param("tenantId") String tenantId,
            @Param("orderId") String orderId
    );

    List<Order> selectByTenantIdAndUserId(
            @Param("tenantId") String tenantId,
            @Param("userId") String userId
    );

    List<Order> selectAllByTenantId(@Param("tenantId") String tenantId);

    int updateRefundState(
            @Param("tenantId") String tenantId,
            @Param("orderId") String orderId,
            @Param("paymentStatus") String paymentStatus,
            @Param("refundAmount") BigDecimal refundAmount,
            @Param("refundedAt") LocalDateTime refundedAt,
            @Param("refundReason") String refundReason,
            @Param("latestEvent") String latestEvent,
            @Param("updatedAt") Instant updatedAt
    );

    int updateCancelState(
            @Param("tenantId") String tenantId,
            @Param("orderId") String orderId,
            @Param("orderStatus") String orderStatus,
            @Param("canceledAt") LocalDateTime canceledAt,
            @Param("cancelReason") String cancelReason,
            @Param("latestEvent") String latestEvent,
            @Param("updatedAt") Instant updatedAt
    );

    int insertOrderEvent(
            @Param("tenantId") String tenantId,
            @Param("eventId") String eventId,
            @Param("orderId") String orderId,
            @Param("eventType") String eventType,
            @Param("eventPayload") String eventPayload,
            @Param("operatorType") String operatorType,
            @Param("operatorId") String operatorId,
            @Param("traceId") String traceId,
            @Param("idempotencyKey") String idempotencyKey,
            @Param("requestFingerprint") String requestFingerprint,
            @Param("createdAt") Instant createdAt
    );

    OrderEvent selectEventByTenantIdAndIdempotencyKey(
            @Param("tenantId") String tenantId,
            @Param("idempotencyKey") String idempotencyKey
    );
}
