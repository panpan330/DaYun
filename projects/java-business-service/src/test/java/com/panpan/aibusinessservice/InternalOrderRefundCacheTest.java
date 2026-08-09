package com.panpan.aibusinessservice;

import static com.panpan.aibusinessservice.InternalApiTestSupport.withInternalHeaders;
import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.common.cache.OrderCache;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.entity.Order;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

/**
 * 验证退款写库后订单缓存被刷新（fix 2）：退款前缓存已有 paid 快照时，
 * queryOrder / 幂等返回不再读到退款前的 stale 快照。
 * 内存缓存以「副本」语义存储（模拟 Redis 序列化快照，而非对象引用），
 * 使该测试能真实区分「applyRefund 显式刷新缓存」与「对象引用共享」。
 */
@SpringBootTest
@AutoConfigureMockMvc
@Transactional
class InternalOrderRefundCacheTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private OrderCache orderCache;

    @BeforeEach
    void clearCache() {
        ((InMemoryOrderCache) orderCache).clear();
    }

    @Test
    void refundRefreshesOrderCacheAndQuerySeesRefunded() throws Exception {
        // 1. 退款前先查一次：把 paid 快照写入缓存（模拟生产：缓存中已有旧快照）
        mockMvc.perform(withInternalHeaders(get("/internal/orders/A1002")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.payment_status").value("paid"));

        // 2. 退款
        mockMvc.perform(withInternalHeaders(post("/internal/orders/A1002/refund"))
                        .header(TraceHeaders.IDEMPOTENCY_KEY, "refund-stage7-cache-001")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"reason\": \"缓存刷新验证\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.payment_status").value("refunded"));

        // 3. 缓存中的快照已被刷新为 refunded（修复前这里读到的是 stale 的 paid 副本）
        Order cached = orderCache.get("default", "A1002").orElseThrow();
        assertThat(cached.getPaymentStatus()).isEqualTo("refunded");
        assertThat(cached.getRefundAmount()).isEqualByComparingTo("159.00");
        assertThat(cached.getRefundReason()).isEqualTo("缓存刷新验证");

        // 4. queryOrder 读到 refunded，不再是 stale 的 paid 快照
        mockMvc.perform(withInternalHeaders(get("/internal/orders/A1002")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.payment_status").value("refunded"))
                .andExpect(jsonPath("$.data.refund_amount").value(closeTo(159.00, 0.001)));
    }

    @TestConfiguration
    static class CacheConfig {
        @Bean
        @Primary
        OrderCache inMemoryOrderCache(ObjectMapper objectMapper) {
            return new InMemoryOrderCache(objectMapper);
        }
    }

    static class InMemoryOrderCache implements OrderCache {
        private final Map<String, Order> store = new ConcurrentHashMap<>();
        private final ObjectMapper objectMapper;

        InMemoryOrderCache(ObjectMapper objectMapper) {
            this.objectMapper = objectMapper;
        }

        void clear() {
            store.clear();
        }

        @Override
        public Optional<Order> get(String tenantId, String orderId) {
            return Optional.ofNullable(store.get(tenantId + ":" + orderId)).map(this::copy);
        }

        @Override
        public void put(Order order) {
            store.put(order.getTenantId() + ":" + order.getOrderId(), copy(order));
        }

        private Order copy(Order order) {
            try {
                return objectMapper.readValue(objectMapper.writeValueAsString(order), Order.class);
            } catch (JsonProcessingException exception) {
                throw new IllegalStateException("InMemoryOrderCache copy failed", exception);
            }
        }
    }
}
