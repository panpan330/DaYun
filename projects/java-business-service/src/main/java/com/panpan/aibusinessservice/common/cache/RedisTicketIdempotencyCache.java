package com.panpan.aibusinessservice.common.cache;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import com.panpan.aibusinessservice.common.redis.RedisKeys;
import java.time.Duration;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "true")
public class RedisTicketIdempotencyCache implements TicketIdempotencyCache {
    private static final Logger log = LoggerFactory.getLogger(RedisTicketIdempotencyCache.class);

    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final RedisFeatureProperties properties;

    public RedisTicketIdempotencyCache(
            StringRedisTemplate redisTemplate,
            ObjectMapper objectMapper,
            RedisFeatureProperties properties
    ) {
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    @Override
    public Optional<TicketIdempotencyCacheEntry> get(String tenantId, String idempotencyKey) {
        String key = RedisKeys.ticketIdempotencyKey(properties.keyPrefix(), tenantId, idempotencyKey);
        try {
            String raw = redisTemplate.opsForValue().get(key);
            if (raw == null || raw.isBlank()) {
                return Optional.empty();
            }
            return Optional.of(objectMapper.readValue(raw, TicketIdempotencyCacheEntry.class));
        } catch (RuntimeException | JsonProcessingException exception) {
            log.warn("Redis ticket idempotency cache read failed, key={}, reason={}", key, exception.toString());
            return Optional.empty();
        }
    }

    @Override
    public void put(String tenantId, String idempotencyKey, TicketIdempotencyCacheEntry entry) {
        String key = RedisKeys.ticketIdempotencyKey(properties.keyPrefix(), tenantId, idempotencyKey);
        try {
            String raw = objectMapper.writeValueAsString(entry);
            redisTemplate.opsForValue().set(
                    key,
                    raw,
                    Duration.ofSeconds(properties.ticketIdempotencyTtlSeconds())
            );
        } catch (RuntimeException | JsonProcessingException exception) {
            log.warn("Redis ticket idempotency cache write failed, key={}, reason={}", key, exception.toString());
        }
    }
}
