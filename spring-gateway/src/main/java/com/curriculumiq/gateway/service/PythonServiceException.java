package com.curriculumiq.gateway.service;

import org.springframework.http.HttpStatus;

/** Safe, user-facing failure when the Python AI service cannot be reached or errors. */
public class PythonServiceException extends RuntimeException {
    private final HttpStatus status;

    public PythonServiceException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public HttpStatus getStatus() {
        return status;
    }
}
