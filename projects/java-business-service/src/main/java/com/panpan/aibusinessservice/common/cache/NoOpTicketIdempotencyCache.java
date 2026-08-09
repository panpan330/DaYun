package com.panpan.aibusinessservice.common.cache;

import java.util.Optional;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "false")
public class NoOpTicketIdempotencyCache implements TicketIdempotencyCache {
    @Override
    public Optional<TicketIdempotencyCacheEntry> get(String tenantId, String idempotencyKey) {
        return Optional.empty();
    }

    @Override
    public void put(String tenantId, String idempotencyKey, TicketIdempotencyCacheEntry entry) {
    }
}
