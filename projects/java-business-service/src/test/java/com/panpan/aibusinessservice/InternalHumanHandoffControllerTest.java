package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.time.Instant;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class InternalHumanHandoffControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTable() {
        jdbcTemplate.update("DELETE FROM ai_human_handoffs");
    }

    private org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder withInternalHeaders(
            org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder request
    ) {
        return InternalApiTestSupport.withInternalHeaders(request);
    }

    private void insertHandoff(long id, String conversationId, String status, String assignedAgent) {
        jdbcTemplate.update(
                "INSERT INTO ai_human_handoffs "
                        + "(id, conversation_id, user_id, tenant_id, reason, status, assigned_agent, created_at, updated_at, resolved_at) "
                        + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                id, conversationId, "U1001", "default", "愤怒情绪转人工", status, assignedAgent,
                Instant.now(), Instant.now(), status.equals("closed") ? Instant.now() : null);
    }

    @Test
    void activeByConversation_returnsActiveHandoff() throws Exception {
        insertHandoff(1L, "conv-1", "in_progress", "A1001");

        mockMvc.perform(withInternalHeaders(get("/internal/ai-human-handoffs/active-by-conversation")
                        .param("conversationId", "conv-1")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data.conversation_id").value("conv-1"))
                .andExpect(jsonPath("$.data.status").value("in_progress"))
                .andExpect(jsonPath("$.data.assigned_agent").value("A1001"));
    }

    @Test
    void activeByConversation_returnsNullWhenNone() throws Exception {
        mockMvc.perform(withInternalHeaders(get("/internal/ai-human-handoffs/active-by-conversation")
                        .param("conversationId", "conv-none")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data").doesNotExist());
    }

    @Test
    void activeByConversation_excludesClosed() throws Exception {
        insertHandoff(2L, "conv-2", "closed", "A1001");

        mockMvc.perform(withInternalHeaders(get("/internal/ai-human-handoffs/active-by-conversation")
                        .param("conversationId", "conv-2")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data").doesNotExist());
    }
}
