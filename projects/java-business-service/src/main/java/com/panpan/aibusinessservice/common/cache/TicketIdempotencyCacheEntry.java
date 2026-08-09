package com.panpan.aibusinessservice.common.cache;

public record TicketIdempotencyCacheEntry(
        String requestFingerprint,
        String ticketId
) {
}
