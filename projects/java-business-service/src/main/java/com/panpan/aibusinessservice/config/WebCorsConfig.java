package com.panpan.aibusinessservice.config;

import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebCorsConfig implements WebMvcConfigurer {
    private final List<String> allowedOriginPatterns;

    public WebCorsConfig(
            @Value("#{'${app.cors.allowed-origin-patterns:http://127.0.0.1:*,http://localhost:*}'.split(',')}")
            List<String> allowedOriginPatterns
    ) {
        this.allowedOriginPatterns = allowedOriginPatterns.stream()
                .map(String::trim)
                .filter(origin -> !origin.isBlank())
                .toList();
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns(allowedOriginPatterns.toArray(String[]::new))
                .allowedMethods("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .exposedHeaders("X-Trace-Id")
                .allowCredentials(false)
                .maxAge(3600);
    }
}
