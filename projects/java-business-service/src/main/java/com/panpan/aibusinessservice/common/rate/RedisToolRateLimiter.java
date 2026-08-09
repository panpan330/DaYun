package com.panpan.aibusinessservice.common.rate;

import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import com.panpan.aibusinessservice.common.redis.RedisKeys;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import java.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "true")
public class RedisToolRateLimiter implements ToolRateLimiter {
    private static final Logger log = LoggerFactory.getLogger(RedisToolRateLimiter.class);

    private final StringRedisTemplate redisTemplate;
    private final RedisFeatureProperties properties;

    public RedisToolRateLimiter(StringRedisTemplate redisTemplate, RedisFeatureProperties properties) {
        this.redisTemplate = redisTemplate;
        this.properties = properties;
    }

    @Override
    public void check(InternalRequestContext context, String method, String uri) {
        RedisFeatureProperties.RateLimit rateLimit = properties.rateLimit();
        if (!rateLimit.enabled()) {
            return;
        }

        String key = RedisKeys.rateLimitKey(
                properties.keyPrefix(),
                context.tenantId(),
                context.userId(),
                method,
                uri
        );
        try {
            Long count = redisTemplate.opsForValue().increment(key);
            if (count != null && count == 1L) {
                redisTemplate.expire(key, Duration.ofSeconds(rateLimit.windowSeconds()));
            }
            if (count != null && count > rateLimit.limit()) {
                throw new BusinessException(BusinessErrorCode.TOOL_RATE_LIMITED);
            }
        } catch (BusinessException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.warn("Redis tool rate limit check failed, key={}, reason={}", key, exception.toString());
        }
    }
}
