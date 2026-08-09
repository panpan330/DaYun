package com.panpan.aibusinessservice.config;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Configuration;

@Configuration
@MapperScan("com.panpan.aibusinessservice.mapper")
public class MyBatisConfig {
}
