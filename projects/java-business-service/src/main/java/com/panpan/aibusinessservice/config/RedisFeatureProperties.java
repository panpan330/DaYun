package com.panpan.aibusinessservice.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.redis")
public record RedisFeatureProperties(
        boolean enabled,
        String keyPrefix,
        long orderCacheTtlSeconds,
        long ticketIdempotencyTtlSeconds,
        RateLimit rateLimit
) {
    public RedisFeatureProperties {
        if (keyPrefix == null || keyPrefix.isBlank()) {
            keyPrefix = "java-business";
        }
        if (orderCacheTtlSeconds <= 0) {
            orderCacheTtlSeconds = 300;
        }
        if (ticketIdempotencyTtlSeconds <= 0) {
            ticketIdempotencyTtlSeconds = 86400;
        }
        if (rateLimit == null) {
            rateLimit = new RateLimit(false, 60, 60);
        }
    }

    public record RateLimit(
            boolean enabled,
            int limit,
            long windowSeconds
    ) {
        public RateLimit {
            if (limit <= 0) {
                limit = 60;
            }
            if (windowSeconds <= 0) {
                windowSeconds = 60;
            }
        }
    }
}
