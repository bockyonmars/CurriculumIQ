package com.curriculumiq.gateway.web;

import com.curriculumiq.gateway.dto.DTOs.QuestionRequest;
import com.curriculumiq.gateway.dto.DTOs.QuestionResponse;
import com.curriculumiq.gateway.service.PythonServiceClient;
import com.curriculumiq.gateway.service.PythonServiceException;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class QuestionController {

    private final PythonServiceClient python;

    public QuestionController(PythonServiceClient python) {
        this.python = python;
    }

    /** Accept a document_id + question and forward to the Python service. */
    @PostMapping("/questions")
    public QuestionResponse ask(@RequestBody QuestionRequest request) {
        if (request == null || request.document_id() == null || request.document_id().isBlank()
                || request.question() == null || request.question().isBlank()) {
            throw new PythonServiceException(HttpStatus.BAD_REQUEST,
                    "A document_id and a question are required.");
        }
        return python.forwardQuestion(request);
    }
}
