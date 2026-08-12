package com.curriculumiq.gateway.config;

import java.time.Duration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

/**
 * Builds the {@link RestClient} used to reach the Python AI service, with
 * externalized base URL and connect/read timeouts.
 */
@Configuration
public class RestClientConfig {

    @Bean
    public RestClient pythonServiceRestClient(
            @Value("${python.service.url:http://localhost:8000}") String baseUrl,
            @Value("${python.service.connect-timeout-ms:3000}") long connectMs,
            @Value("${python.service.read-timeout-ms:60000}") long readMs) {

        var settings = ClientHttpRequestFactorySettings.DEFAULTS
                .withConnectTimeout(Duration.ofMillis(connectMs))
                .withReadTimeout(Duration.ofMillis(readMs));

        return RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(ClientHttpRequestFactories.get(settings))
                .build();
    }
}
