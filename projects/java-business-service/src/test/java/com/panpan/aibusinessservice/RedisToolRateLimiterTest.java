package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.panpan.aibusinessservice.common.rate.RedisToolRateLimiter;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import java.time.Duration;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

class RedisToolRateLimiterTest {
    private final StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
    @SuppressWarnings("unchecked")
    private final ValueOperations<String, String> valueOperations = mock(ValueOperations.class);
    private final RedisFeatureProperties properties = new RedisFeatureProperties(
            true,
            "test-java-business",
            300,
            86400,
            new RedisFeatureProperties.RateLimit(true, 2, 60)
    );
    private final RedisToolRateLimiter rateLimiter = new RedisToolRateLimiter(redisTemplate, properties);
    private final InternalRequestContext context = new InternalRequestContext(
            "trace-test",
            "ai-service",
            "user-001",
            "tenant-a"
    );

    @Test
    void firstRequestCreatesWindowTtl() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.increment(anyString())).thenReturn(1L);

        rateLimiter.check(context, "GET", "/internal/orders/A1001");

        verify(redisTemplate).expire(anyString(), any(Duration.class));
    }

    @Test
    void requestOverLimitIsRejected() {
        when(redisTemplate.opsForValue()).thenReturn(valueOperations);
        when(valueOperations.increment(anyString())).thenReturn(3L);

        assertThatThrownBy(() -> rateLimiter.check(context, "GET", "/internal/orders/A1001"))
                .isInstanceOf(BusinessException.class)
                .satisfies(exception -> assertThat(((BusinessException) exception).errorCode())
                        .isEqualTo(BusinessErrorCode.TOOL_RATE_LIMITED));
    }

    @Test
    void redisFailureDoesNotBreakBusinessRequest() {
        when(redisTemplate.opsForValue()).thenThrow(new IllegalStateException("redis down"));

        rateLimiter.check(context, "GET", "/internal/orders/A1001");
    }
}
