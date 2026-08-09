package com.panpan.aibusinessservice.common.redis;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public final class RedisKeys {
    private RedisKeys() {
    }

    public static String orderCacheKey(String prefix, String tenantId, String orderId) {
        return join(prefix, "order", tenantId, orderId);
    }

    public static String ticketIdempotencyKey(String prefix, String tenantId, String idempotencyKey) {
        return join(prefix, "ticket-idempotency", tenantId, idempotencyKey);
    }

    public static String rateLimitKey(String prefix, String tenantId, String userId, String method, String uri) {
        return join(prefix, "rate-limit", tenantId, userId, method, uri);
    }

    private static String join(String prefix, String category, String... parts) {
        StringBuilder key = new StringBuilder(safe(prefix)).append(':').append(category);
        for (String part : parts) {
            key.append(':').append(safe(part));
        }
        return key.toString();
    }

    private static String safe(String value) {
        return URLEncoder.encode(value == null ? "" : value, StandardCharsets.UTF_8);
    }
}
