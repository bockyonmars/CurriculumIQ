package com.curriculumiq.gateway;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import com.curriculumiq.gateway.dto.DTOs.Citation;
import com.curriculumiq.gateway.dto.DTOs.QuestionResponse;
import com.curriculumiq.gateway.service.PythonServiceClient;
import com.curriculumiq.gateway.service.PythonServiceException;
import com.curriculumiq.gateway.web.GlobalExceptionHandler;
import com.curriculumiq.gateway.web.QuestionController;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(QuestionController.class)
@Import(GlobalExceptionHandler.class)
class QuestionControllerTest {

    @Autowired
    private MockMvc mvc;

    @MockBean
    private PythonServiceClient python;

    @Test
    void returnsAnswerAndCitations() throws Exception {
        when(python.forwardQuestion(any())).thenReturn(new QuestionResponse(
                "A quadratic has form ax^2+bx+c [S1].", false,
                List.of(new Citation("S1", "intro_to_algebra.pdf", 5, "A quadratic..."))));

        mvc.perform(post("/api/questions").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"document_id\":\"doc_1\",\"question\":\"What is a quadratic?\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.abstained").value(false))
                .andExpect(jsonPath("$.citations[0].page").value(5))
                .andExpect(jsonPath("$.citations[0].filename").value("intro_to_algebra.pdf"));
    }

    @Test
    void missingFieldsRejected() throws Exception {
        mvc.perform(post("/api/questions").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"document_id\":\"\",\"question\":\"\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.detail").exists());
    }

    @Test
    void downstreamUnavailableIsSafe() throws Exception {
        when(python.forwardQuestion(any())).thenThrow(new PythonServiceException(
                HttpStatus.SERVICE_UNAVAILABLE, "The AI service is currently unavailable."));

        mvc.perform(post("/api/questions").contentType(MediaType.APPLICATION_JSON)
                        .content("{\"document_id\":\"d\",\"question\":\"q\"}"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.detail").value("The AI service is currently unavailable."));
    }
}
