package com.curriculumiq.gateway;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.curriculumiq.gateway.service.PythonServiceClient;
import com.curriculumiq.gateway.web.HealthController;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

@WebMvcTest(HealthController.class)
class HealthControllerTest {

    @Autowired
    private MockMvc mvc;

    @MockBean
    private PythonServiceClient python;

    @Test
    void reportsGatewayAndPythonStatus() throws Exception {
        when(python.isPythonHealthy()).thenReturn(true);
        mvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.gateway").value("ok"))
                .andExpect(jsonPath("$.pythonService").value("up"));
    }

    @Test
    void reportsPythonDown() throws Exception {
        when(python.isPythonHealthy()).thenReturn(false);
        mvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.pythonService").value("down"));
    }
}
