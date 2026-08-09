package com.panpan.aibusinessservice;

import static com.panpan.aibusinessservice.InternalApiTestSupport.withInternalHeaders;
import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
class AiResponseFeedbackControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void clearFeedback() {
        jdbcTemplate.update("DELETE FROM ai_response_feedback");
    }

    @Test
    void internalFeedbackUpsertsOneResponseIdentity() throws Exception {
        MvcResult first = mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(unhelpfulFeedbackJson())
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.rating").value("unhelpful"))
                .andExpect(jsonPath("$.data.reason").value("citation_irrelevant"))
                .andReturn();

        long feedbackId = objectMapper.readTree(first.getResponse().getContentAsString())
                .path("data").path("feedback_id").asLong();

        mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(unhelpfulFeedbackJson()
                                        .replace("unhelpful", "helpful")
                                        .replace("\"citation_irrelevant\"", "null"))
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.feedback_id").value(feedbackId))
                .andExpect(jsonPath("$.data.rating").value("helpful"))
                .andExpect(jsonPath("$.data.reason").doesNotExist());

        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ai_response_feedback WHERE conversation_id = ? AND trace_id = ?",
                Integer.class,
                "agent-feedback-001",
                "trace-feedback-001"
        );
        assertThat(count).isEqualTo(1);
    }

    @Test
    void onlySupervisorCanViewTenantFeedbackOverview() throws Exception {
        mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(unhelpfulFeedbackJson())
                )
                .andExpect(status().isOk());

        String customerToken = loginAndExtractToken("customer");
        mockMvc.perform(
                        get("/api/ai-response-feedback/overview")
                                .header(TraceHeaders.TRACE_ID, "trace-feedback-customer")
                                .header("Authorization", "Bearer " + customerToken)
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("AI_FEEDBACK_ACCESS_DENIED"));

        String supervisorToken = loginAndExtractToken("supervisor");
        mockMvc.perform(
                        get("/api/ai-response-feedback/overview")
                                .header(TraceHeaders.TRACE_ID, "trace-feedback-supervisor")
                                .header("Authorization", "Bearer " + supervisorToken)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.unhelpful_count").value(1))
                .andExpect(jsonPath("$.data.regression_candidates[0].trace_id").value("trace-feedback-001"));
    }

    @Test
    void internalPromotionMarksCandidateAsRegressionAdded() throws Exception {
        MvcResult created = mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(unhelpfulFeedbackJson())
                )
                .andExpect(status().isOk())
                .andReturn();
        long feedbackId = objectMapper.readTree(created.getResponse().getContentAsString())
                .path("data").path("feedback_id").asLong();

        mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback/{feedbackId}/promote", feedbackId))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"bad_case_id\":\"bad_production_feedback_stage11_feedback_1_agent_decision\"}")
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.review_status").value("regression_added"))
                .andExpect(jsonPath("$.data.bad_case_id").value("bad_production_feedback_stage11_feedback_1_agent_decision"));
    }

    @Test
    void internalReviewCanTriagedCandidateWithNote() throws Exception {
        MvcResult created = mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(unhelpfulFeedbackJson())
                )
                .andExpect(status().isOk())
                .andReturn();
        long feedbackId = objectMapper.readTree(created.getResponse().getContentAsString())
                .path("data").path("feedback_id").asLong();

        mockMvc.perform(
                        withInternalHeaders(post("/internal/ai-response-feedback/{feedbackId}/review", feedbackId))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"review_status\":\"triaged\",\"review_note\":\"Need policy review.\"}")
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.review_status").value("triaged"))
                .andExpect(jsonPath("$.data.review_note").value("Need policy review."));
    }

    private String loginAndExtractToken(String username) throws Exception {
        MvcResult result = mockMvc.perform(
                        post("/api/auth/login")
                                .header(TraceHeaders.TRACE_ID, "trace-feedback-login")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {"username":"%s","password":"123456","tenant_id":"default"}
                                        """.formatted(username))
                )
                .andExpect(status().isOk())
                .andReturn();
        JsonNode body = objectMapper.readTree(result.getResponse().getContentAsString());
        return body.path("data").path("token").asText();
    }

    private String unhelpfulFeedbackJson() {
        return """
                {
                  "conversation_id":"agent-feedback-001",
                  "trace_id":"trace-feedback-001",
                  "rating":"unhelpful",
                  "reason":"citation_irrelevant",
                  "agent_route":"policy_rag",
                  "citation_count":2,
                  "human_handoff_suggested":false
                }
                """;
    }
}
