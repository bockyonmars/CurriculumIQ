package com.curriculumiq.gateway.service;

import java.io.IOException;

import com.curriculumiq.gateway.dto.DTOs.DocumentResponse;
import com.curriculumiq.gateway.dto.DTOs.ErrorResponse;
import com.curriculumiq.gateway.dto.DTOs.QuestionRequest;
import com.curriculumiq.gateway.dto.DTOs.QuestionResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;

/**
 * Forwards requests to the Python AI service. Converts transport/HTTP failures
 * into a safe {@link PythonServiceException} — never leaking stack traces.
 */
@Service
public class PythonServiceClient {

    private static final Logger log = LoggerFactory.getLogger(PythonServiceClient.class);

    private final RestClient client;

    public PythonServiceClient(RestClient pythonServiceRestClient) {
        this.client = pythonServiceRestClient;
    }

    /** True if the Python service reports healthy; false on any failure. */
    public boolean isPythonHealthy() {
        try {
            client.get().uri("/health").retrieve().toBodilessEntity();
            return true;
        } catch (Exception ex) {
            log.warn("Python service health check failed: {}", ex.getClass().getSimpleName());
            return false;
        }
    }

    public DocumentResponse forwardDocument(String filename, byte[] content) {
        Resource part = new ByteArrayResource(content) {
            @Override
            public String getFilename() {
                return filename != null ? filename : "upload.pdf";
            }
        };
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", part);

        long start = System.currentTimeMillis();
        try {
            DocumentResponse response = client.post()
                    .uri("/api/documents")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, this::raise)
                    .body(DocumentResponse.class);
            log.info("proxied POST /api/documents -> ready ({} ms)", System.currentTimeMillis() - start);
            return response;
        } catch (PythonServiceException ex) {
            throw ex;
        } catch (Exception ex) {
            throw unreachable(ex);
        }
    }

    public QuestionResponse forwardQuestion(QuestionRequest request) {
        long start = System.currentTimeMillis();
        try {
            QuestionResponse response = client.post()
                    .uri("/api/questions")
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(request)
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, this::raise)
                    .body(QuestionResponse.class);
            log.info("proxied POST /api/questions -> answered ({} ms)", System.currentTimeMillis() - start);
            return response;
        } catch (PythonServiceException ex) {
            throw ex;
        } catch (Exception ex) {
            throw unreachable(ex);
        }
    }

    /** Map an error response body into a safe exception without leaking internals. */
    private void raise(org.springframework.http.HttpRequest req,
                       org.springframework.http.client.ClientHttpResponse res) throws IOException {
        HttpStatusCode code = res.getStatusCode();
        String detail = "The AI service could not process this request.";
        try {
            byte[] bytes = res.getBody().readAllBytes();
            if (bytes.length > 0) {
                ErrorResponse parsed = new com.fasterxml.jackson.databind.ObjectMapper()
                        .readValue(bytes, ErrorResponse.class);
                if (parsed != null && parsed.detail() != null && !parsed.detail().isBlank()) {
                    detail = parsed.detail();
                }
            }
        } catch (Exception ignored) {
            // Keep the generic safe message.
        }
        HttpStatus status = HttpStatus.resolve(code.value());
        throw new PythonServiceException(status != null ? status : HttpStatus.BAD_GATEWAY, detail);
    }

    private PythonServiceException unreachable(Exception ex) {
        log.warn("Python service unreachable: {}", ex.getClass().getSimpleName());
        return new PythonServiceException(HttpStatus.SERVICE_UNAVAILABLE,
                "The AI service is currently unavailable. Please try again shortly.");
    }
}
