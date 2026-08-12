package com.curriculumiq.gateway.dto;

import java.util.List;

/**
 * DTOs for the gateway's public API and the Python service contract.
 * Kept as records — the gateway forwards safe, student-facing fields only and
 * never surfaces keys, prompts, internal paths, or distances.
 */
public final class DTOs {
    private DTOs() {}

    public record HealthResponse(String gateway, String pythonService) {}

    public record DocumentResponse(String document_id, String filename, int pages,
                                   int chunks, List<Integer> skipped_pages, String status) {}

    public record QuestionRequest(String document_id, String question) {}

    public record Citation(String source_id, String filename, int page, String passage) {}

    public record QuestionResponse(String answer, boolean abstained, List<Citation> citations) {}

    public record ErrorResponse(String detail) {}
}
