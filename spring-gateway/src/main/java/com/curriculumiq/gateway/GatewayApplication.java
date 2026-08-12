package com.curriculumiq.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * CurriculumIQ Spring gateway. Proxies the Python AI service; it never calls
 * OpenAI directly and holds no OpenAI key.
 */
@SpringBootApplication
public class GatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
