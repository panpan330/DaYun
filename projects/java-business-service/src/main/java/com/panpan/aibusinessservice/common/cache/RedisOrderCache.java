package com.panpan.aibusinessservice.common.cache;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import com.panpan.aibusinessservice.common.redis.RedisKeys;
import com.panpan.aibusinessservice.entity.Order;
import java.time.Duration;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "true")
public class RedisOrderCache implements OrderCache {
    private static final Logger log = LoggerFactory.getLogger(RedisOrderCache.class);

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final RedisFeatureProperties properties;

    public RedisOrderCache(
            StringRedisTemplate redisTemplate,
            ObjectMapper objectMapper,
            RedisFeatureProperties properties
    ) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    @Override
    public Optional<Order> get(String tenantId, String orderId) {
        String key = RedisKeys.orderCacheKey(properties.keyPrefix(), tenantId, orderId);
        try {
            String raw = redisTemplate.opsForValue().get(key);
            if (raw == null || raw.isBlank()) {
                return Optional.empty();
            }
            CachedOrder cached = objectMapper.readValue(raw, CachedOrder.class);
            return Optional.of(cached.toOrder());
        } catch (RuntimeException | JsonProcessingException exception) {
            log.warn("Redis order cache read failed, key={}, reason={}", key, exception.toString());
            return Optional.empty();
        }
    }

    @Override
    public void put(Order order) {
        String key = RedisKeys.orderCacheKey(properties.keyPrefix(), order.getTenantId(), order.getOrderId());
        try {
            String raw = objectMapper.writeValueAsString(CachedOrder.from(order));
            redisTemplate.opsForValue().set(
                    key,
                    raw,
                    Duration.ofSeconds(properties.orderCacheTtlSeconds())
            );
        } catch (RuntimeException | JsonProcessingException exception) {
            log.warn("Redis order cache write failed, key={}, reason={}", key, exception.toString());
        }
    }

    private record CachedOrder(
            String orderId,
            String ownerUserId,
            String tenantId,
            String orderStatus,
            String paymentStatus,
            String logisticsMessage,
            String latestEvent,
            boolean canCreateTicket,
            java.math.BigDecimal amount,
            java.math.BigDecimal refundAmount,
            java.time.LocalDateTime refundedAt,
            String refundReason
    ) {
        static CachedOrder from(Order order) {
            return new CachedOrder(
                    order.getOrderId(),
                    order.getOwnerUserId(),
                    order.getTenantId(),
                    order.getOrderStatus(),
                    order.getPaymentStatus(),
                    order.getLogisticsMessage(),
                    order.getLatestEvent(),
                    order.isCanCreateTicket(),
                    order.getAmount(),
                    order.getRefundAmount(),
                    order.getRefundedAt(),
                    order.getRefundReason()
            );
        }

        Order toOrder() {
            Order order = new Order(
                    orderId,
                    ownerUserId,
                    tenantId,
                    orderStatus,
                    paymentStatus,
                    logisticsMessage,
                    latestEvent,
                    canCreateTicket
            );
            order.setAmount(amount);
            order.setRefundAmount(refundAmount);
            order.setRefundedAt(refundedAt);
            order.setRefundReason(refundReason);
            return order;
        }
    }
}
