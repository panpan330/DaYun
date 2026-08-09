package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.options;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
class AuthAndKnowledgeControllerTest {
    private static final String TRACE_ID = "trace-stage11-auth-test";

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void loginPreflightAllowsLocalFrontendOrigin() throws Exception {
        mockMvc.perform(
                        options("/api/auth/login")
                                .header("Origin", "http://127.0.0.1:5173")
                                .header("Access-Control-Request-Method", "POST")
                                .header("Access-Control-Request-Headers", "content-type")
                )
                .andExpect(status().isOk())
                .andExpect(header().string("Access-Control-Allow-Origin", "http://127.0.0.1:5173"))
                .andExpect(header().string("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS"));
    }

    @Test
    void loginReturnsCurrentUserAndLocalDevToken() throws Exception {
        mockMvc.perform(
                        post("/api/auth/login")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "username": "agent",
                                          "password": "123456"
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(header().string(TraceHeaders.TRACE_ID, TRACE_ID))
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.code").value("OK"))
                .andExpect(jsonPath("$.data.token").exists())
                .andExpect(jsonPath("$.data.user.user_id").value("A1001"))
                .andExpect(jsonPath("$.data.user.roles[0]").value("agent"))
                .andExpect(jsonPath("$.data.user.default_home_path").value("/workbench"));
    }

    @Test
    void loginRejectsWrongPassword() throws Exception {
        mockMvc.perform(
                        post("/api/auth/login")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "username": "agent",
                                          "password": "wrong-password"
                                        }
                                        """)
                )
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("LOGIN_FAILED"));
    }

    @Test
    void meReturnsCurrentUserFromAuthorizationHeader() throws Exception {
        String token = loginAndExtractToken("supervisor");

        mockMvc.perform(
                        get("/api/auth/me")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.user_id").value("S1001"))
                .andExpect(jsonPath("$.data.roles[0]").value("supervisor"))
                .andExpect(jsonPath("$.data.default_home_path").value("/knowledge"));
    }

    @Test
    void meRejectsMissingAuthorizationHeader() throws Exception {
        mockMvc.perform(
                        get("/api/auth/me")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                )
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("AUTH_REQUIRED"));
    }

    @Test
    void knowledgeDocumentsAreFilteredByCurrentUserRole() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        String customerToken = loginAndExtractToken("customer");

        mockMvc.perform(
                        get("/api/knowledge-documents")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + agentToken)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.length()").value(3))
                .andExpect(jsonPath("$.data[0].permission_group").value("customer_service"));

        MvcResult customerResult = mockMvc.perform(
                        get("/api/knowledge-documents")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + customerToken)
                )
                .andExpect(status().isOk())
                .andReturn();

        JsonNode body = objectMapper.readTree(customerResult.getResponse().getContentAsString());
        assertThat(body.path("data")).isEmpty();
    }

    private String loginAndExtractToken(String username) throws Exception {
        MvcResult result = mockMvc.perform(
                        post("/api/auth/login")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "username": "%s",
                                          "password": "123456"
                                        }
                                        """.formatted(username))
                )
                .andExpect(status().isOk())
                .andReturn();

        JsonNode body = objectMapper.readTree(result.getResponse().getContentAsString());
        return body.path("data").path("token").asText();
    }
}
