package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.config.InternalApiProperties;
import com.panpan.aibusinessservice.config.RedisFeatureProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableConfigurationProperties({InternalApiProperties.class, RedisFeatureProperties.class})
@EnableScheduling
public class AiBusinessServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiBusinessServiceApplication.class, args);
    }
}
