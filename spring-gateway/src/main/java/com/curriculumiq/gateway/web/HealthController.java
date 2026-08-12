package com.curriculumiq.gateway.web;

import com.curriculumiq.gateway.dto.DTOs.HealthResponse;
import com.curriculumiq.gateway.service.PythonServiceClient;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class HealthController {

    private final PythonServiceClient python;

    public HealthController(PythonServiceClient python) {
        this.python = python;
    }

    /** Gateway status plus the downstream Python service availability. */
    @GetMapping("/health")
    public HealthResponse health() {
        return new HealthResponse("ok", python.isPythonHealthy() ? "up" : "down");
    }
}
