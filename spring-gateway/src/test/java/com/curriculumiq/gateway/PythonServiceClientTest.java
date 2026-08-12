package com.curriculumiq.gateway;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withStatus;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import com.curriculumiq.gateway.dto.DTOs.QuestionRequest;
import com.curriculumiq.gateway.dto.DTOs.QuestionResponse;
import com.curriculumiq.gateway.service.PythonServiceClient;
import com.curriculumiq.gateway.service.PythonServiceException;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

/**
 * Proves the gateway really proxies a (mocked) Python-service response, and
 * maps downstream errors into safe exceptions. No OpenAI, no network.
 */
class PythonServiceClientTest {

    private PythonServiceClient clientWith(RestClient.Builder builder) {
        return new PythonServiceClient(builder.baseUrl("http://python-ai-service:8000").build());
    }

    @Test
    void proxiesQuestionResponse() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo("http://python-ai-service:8000/api/questions"))
              .andExpect(method(HttpMethod.POST))
              .andRespond(withSuccess("""
                  {"answer":"A quadratic has the form ax^2+bx+c [S1].",
                   "abstained":false,
                   "citations":[{"source_id":"S1","filename":"intro_to_algebra.pdf",
                                 "page":5,"passage":"A quadratic function..."}]}
                  """, MediaType.APPLICATION_JSON));

        PythonServiceClient client = clientWith(builder);
        QuestionResponse resp = client.forwardQuestion(
                new QuestionRequest("doc_123", "What is a quadratic?"));

        assertThat(resp.abstained()).isFalse();
        assertThat(resp.citations()).hasSize(1);
        assertThat(resp.citations().get(0).page()).isEqualTo(5);
        assertThat(resp.citations().get(0).filename()).isEqualTo("intro_to_algebra.pdf");
        server.verify();
    }

    @Test
    void mapsDownstreamErrorToSafeException() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo("http://python-ai-service:8000/api/questions"))
              .andRespond(withStatus(HttpStatus.BAD_GATEWAY)
                  .body("{\"detail\":\"The AI service could not process this request.\"}")
                  .contentType(MediaType.APPLICATION_JSON));

        PythonServiceClient client = clientWith(builder);
        assertThatThrownBy(() -> client.forwardQuestion(new QuestionRequest("d", "q")))
                .isInstanceOf(PythonServiceException.class)
                .hasMessageContaining("could not process");
    }

    @Test
    void healthFalseWhenUnreachable() {
        RestClient.Builder builder = RestClient.builder();
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo("http://python-ai-service:8000/health"))
              .andRespond(withStatus(HttpStatus.INTERNAL_SERVER_ERROR));
        PythonServiceClient client = clientWith(builder);
        assertThat(client.isPythonHealthy()).isFalse();
    }
}
