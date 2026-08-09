package com.panpan.aibusinessservice.common.cache;

import java.util.Optional;

public interface TicketIdempotencyCache {
    Optional<TicketIdempotencyCacheEntry> get(String tenantId, String idempotencyKey);

    void put(String tenantId, String idempotencyKey, TicketIdempotencyCacheEntry entry);
}
