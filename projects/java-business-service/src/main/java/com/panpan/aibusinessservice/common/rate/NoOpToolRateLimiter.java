package com.panpan.aibusinessservice.common.rate;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.redis.enabled", havingValue = "false")
public class NoOpToolRateLimiter implements ToolRateLimiter {
    @Override
    public void check(InternalRequestContext context, String method, String uri) {
    }
}
