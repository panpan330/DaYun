package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class InternalAiConversationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_messages");
        jdbcTemplate.update("DELETE FROM ai_conversations");
    }

    private org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder withInternalHeaders(
            org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder request
    ) {
        return InternalApiTestSupport.withInternalHeaders(request);
    }

    private String json(Object value) throws Exception {
        return objectMapper.writeValueAsString(value);
    }

    private Map<String, Object> conversationBody(String conversationId) {
        return Map.of(
                "conversation_id", conversationId,
                "user_id", "U1001",
                "title", "title-" + conversationId,
                "conversation_status", "active"
        );
    }

    private Map<String, Object> messagesBody(String conversationId) {
        return Map.of(
                "conversation_id", conversationId,
                "messages", List.of(
                        Map.of("message_id", "m-1", "sender_type", "user", "content", "hello", "trace_id", "trace-1"),
                        Map.of("message_id", "m-2", "sender_type", "assistant", "content", "hi", "trace_id", "trace-2")
                )
        );
    }

    @Test
    void upsertConversationSucceedsAndIsIdempotent() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-1"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-1"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ai_conversations WHERE conversation_id = 'conv-1'", Integer.class);
        org.junit.jupiter.api.Assertions.assertEquals(1, count);
    }

    @Test
    void batchWriteMessagesReturnsInsertedCountAndIsIdempotent() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-msg"))))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations/messages"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(messagesBody("conv-msg"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.inserted").value(2));

        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations/messages"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(messagesBody("conv-msg"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.inserted").value(0));

        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ai_messages WHERE conversation_id = 'conv-msg'", Integer.class);
        org.junit.jupiter.api.Assertions.assertEquals(2, count);
    }

    @Test
    void listConversationsOrdersByUpdatedAtDesc() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-old"))))
                .andExpect(status().isOk());
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-new"))))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(get("/internal/ai-conversations?limit=20")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.length()").value(2))
                .andExpect(jsonPath("$.data[0].conversation_id").value("conv-new"))
                .andExpect(jsonPath("$.data[1].conversation_id").value("conv-old"));
    }

    @Test
    void getMessagesReturnsAscendingOrder() throws Exception {
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-asc"))))
                .andExpect(status().isOk());
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations/messages"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(messagesBody("conv-asc"))))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(get("/internal/ai-conversations/conv-asc/messages")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.length()").value(2))
                .andExpect(jsonPath("$.data[0].message_id").value("m-1"))
                .andExpect(jsonPath("$.data[1].message_id").value("m-2"))
                .andExpect(jsonPath("$.data[1].sender_type").value("assistant"));
    }

    @Test
    void cleanupDeletesOnlyOldConversations() throws Exception {
        jdbcTemplate.update(
                "INSERT INTO ai_conversations (tenant_id, conversation_id, user_id, title, conversation_status, created_at, updated_at) "
                        + "VALUES ('default', 'conv-stale', 'U1001', 'old', 'active', ?, ?)",
                java.sql.Timestamp.from(Instant.now().minus(31, ChronoUnit.DAYS)),
                java.sql.Timestamp.from(Instant.now().minus(31, ChronoUnit.DAYS)));
        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-fresh"))))
                .andExpect(status().isOk());

        mockMvc.perform(withInternalHeaders(post("/internal/ai-conversations/cleanup?olderThanDays=30")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.deleted").value(1));

        Integer remaining = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ai_conversations WHERE conversation_id = 'conv-fresh'", Integer.class);
        org.junit.jupiter.api.Assertions.assertEquals(1, remaining);
    }

    @Test
    void rejectsRequestWithoutInternalHeaders() throws Exception {
        mockMvc.perform(post("/internal/ai-conversations")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(json(conversationBody("conv-1"))))
                .andExpect(status().isUnauthorized());
    }
}
