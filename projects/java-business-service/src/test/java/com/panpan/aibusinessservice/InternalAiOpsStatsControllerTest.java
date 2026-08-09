package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

@SpringBootTest
@AutoConfigureMockMvc
class InternalAiOpsStatsControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private JdbcTemplate jdbc;

    @BeforeEach
    void clean() {
        jdbc.update("DELETE FROM ai_response_feedback");
        jdbc.update("DELETE FROM ai_human_handoffs");
        jdbc.update("DELETE FROM ai_messages");
    }

    private MockHttpServletRequestBuilder internal(MockHttpServletRequestBuilder request) {
        return InternalApiTestSupport.withInternalHeaders(request);
    }

    @Test
    void summary_returnsAllFourSections() throws Exception {
        jdbc.update(
                "INSERT INTO ai_response_feedback "
                        + "(tenant_id, user_id, conversation_id, trace_id, rating, agent_route, created_at, updated_at) "
                        + "VALUES ('default', 'U1', 'c1', 't1', 'helpful', 'order_query', '2026-08-07 10:00:00', '2026-08-07 10:00:00')");
        jdbc.update(
                "INSERT INTO ai_human_handoffs "
                        + "(conversation_id, user_id, tenant_id, reason, status, emotion, created_at, updated_at) "
                        + "VALUES ('c1', 'U1', 'default', 'r', 'pending', 'angry', '2026-08-07 10:00:00', '2026-08-07 10:00:00')");
        jdbc.update(
                "INSERT INTO ai_messages (tenant_id, message_id, conversation_id, sender_type, content, trace_id, created_at) "
                        + "VALUES ('default', 'm1', 'c1', 'user', 'hi', 't', '2026-08-07 09:00:00')");

        mockMvc.perform(internal(get("/internal/ai-ops-stats/summary").param("days", "7")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data.feedback.helpful").value(1))
                .andExpect(jsonPath("$.data.feedback.helpful_rate").value(1.0))
                .andExpect(jsonPath("$.data.handoffs.pending").value(1))
                .andExpect(jsonPath("$.data.handoffs.total").value(1))
                .andExpect(jsonPath("$.data.emotion_distribution.angry").value(1))
                .andExpect(jsonPath("$.data.conversation_volume['2026-08-07']").value(1));
    }

    @Test
    void summary_returnsEmptySectionsWhenNoData() throws Exception {
        mockMvc.perform(internal(get("/internal/ai-ops-stats/summary").param("days", "7")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.feedback.helpful").value(0))
                .andExpect(jsonPath("$.data.feedback.helpful_rate").doesNotExist())
                .andExpect(jsonPath("$.data.handoffs.total").value(0));
    }

    @Test
    void summary_rejectsInvalidDays() throws Exception {
        mockMvc.perform(internal(get("/internal/ai-ops-stats/summary").param("days", "5")))
                .andExpect(status().isBadRequest());
    }

    @Test
    void summary_requiresInternalAuth() throws Exception {
        mockMvc.perform(get("/internal/ai-ops-stats/summary").param("days", "7"))
                .andExpect(status().is4xxClientError());
    }
}
