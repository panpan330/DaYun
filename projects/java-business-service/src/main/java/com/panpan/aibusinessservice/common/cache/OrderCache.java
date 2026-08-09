package com.panpan.aibusinessservice.common.cache;

import com.panpan.aibusinessservice.entity.Order;
import java.util.Optional;

public interface OrderCache {
    Optional<Order> get(String tenantId, String orderId);

    void put(Order order);
}
