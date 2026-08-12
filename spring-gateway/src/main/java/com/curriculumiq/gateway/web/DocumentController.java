package com.curriculumiq.gateway.web;

import java.io.IOException;

import com.curriculumiq.gateway.dto.DTOs.DocumentResponse;
import com.curriculumiq.gateway.service.PythonServiceClient;
import com.curriculumiq.gateway.service.PythonServiceException;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class DocumentController {

    private final PythonServiceClient python;

    public DocumentController(PythonServiceClient python) {
        this.python = python;
    }

    /** Accept a PDF and forward it to the Python service for preparation. */
    @PostMapping(value = "/documents", consumes = "multipart/form-data")
    public DocumentResponse prepare(@RequestParam("file") MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new PythonServiceException(HttpStatus.BAD_REQUEST, "No PDF file was provided.");
        }
        try {
            return python.forwardDocument(file.getOriginalFilename(), file.getBytes());
        } catch (IOException ex) {
            throw new PythonServiceException(HttpStatus.BAD_REQUEST, "The uploaded file could not be read.");
        }
    }
}
