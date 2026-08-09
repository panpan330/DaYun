package com.panpan.aibusinessservice.common.cache;

import com.panpan.aibusinessservice.entity.Order;
import java.util.Optional;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "false")
public class NoOpOrderCache implements OrderCache {
    @Override
    public Optional<Order> get(String tenantId, String orderId) {
        return Optional.empty();
    }

    @Override
    public void put(Order order) {
    }
}
