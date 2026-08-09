package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.panpan.aibusinessservice.common.cache.RedisOrderCache;
import com.panpan.aibusinessservice.common.redis.RedisKeys;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import com.panpan.aibusinessservice.entity.Order;
import java.math.BigDecimal;
import java.time.Duration;
import java.time.LocalDateTime;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

class RedisOrderCacheTest {
    private final StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
    @SuppressWarnings("unchecked")
    private final ValueOperations<String, String> valueOperations = mock(ValueOperations.class);
    private final ObjectMapper objectMapper = JsonMapper.builder()
            .addModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
            .build();
    private final RedisFeatureProperties properties = new RedisFeatureProperties(
            true,
            "test-java-business",
            300,
            86400,
            new RedisFeatureProperties.RateLimit(true, 60, 60)
    );
    private final RedisOrderCache cache = new RedisOrderCache(redisTemplate, objectMapper, properties);

    @Test
    void cachedOrderRoundTripPreservesRefundFields() throws Exception {
        Order order = refundedOrder();
        String key = RedisKeys.orderCacheKey("test-java-business", "default", "A1002");
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);

        cache.put(order);

        ArgumentCaptor<String> rawCaptor = ArgumentCaptor.forClass(String.class);
        verify(valueOperations).set(eq(key), rawCaptor.capture(), any(Duration.class));
        String raw = rawCaptor.getValue();

        // put 序列化的快照包含退款字段
        JsonNode json = objectMapper.readTree(raw);
        assertThat(json.get("amount").decimalValue()).isEqualByComparingTo("159.00");
        assertThat(json.get("refundAmount").decimalValue()).isEqualByComparingTo("159.00");
        assertThat(json.get("refundedAt").asText()).isNotBlank();
        assertThat(json.get("refundReason").asText()).isEqualTo("七天无理由退货");

        // get 反序列化回 Order，退款字段完整
        when(valueOperations.get(key)).thenReturn(raw);
        Order restored = cache.get("default", "A1002").orElseThrow();

        assertThat(restored.getAmount()).isEqualByComparingTo("159.00");
        assertThat(restored.getRefundAmount()).isEqualByComparingTo("159.00");
        assertThat(restored.getRefundedAt()).isEqualTo(order.getRefundedAt());
        assertThat(restored.getRefundReason()).isEqualTo("七天无理由退货");
        assertThat(restored.getPaymentStatus()).isEqualTo("refunded");
        assertThat(restored.getLatestEvent()).isEqualTo("退款成功");
        assertThat(restored.isCanCreateTicket()).isTrue();
    }

    private Order refundedOrder() {
        Order order = new Order(
                "A1002",
                "U1001",
                "default",
                "waiting_shipment",
                "refunded",
                "商家已接单，等待仓库发货。",
                "退款成功",
                true
        );
        order.setAmount(new BigDecimal("159.00"));
        order.setRefundAmount(new BigDecimal("159.00"));
        order.setRefundedAt(LocalDateTime.of(2026, 8, 6, 10, 0));
        order.setRefundReason("七天无理由退货");
        return order;
    }
}
