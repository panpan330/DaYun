package com.panpan.aibusinessservice.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.internal")
public record InternalApiProperties(
        String token,
        String allowedCaller
) {
    public InternalApiProperties {
        if (allowedCaller == null || allowedCaller.isBlank()) {
            allowedCaller = "ai-service";
        }
    }
}
