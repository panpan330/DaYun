package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
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
class HumanHandoffControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_human_handoffs");
    }

    private String loginAndExtractToken(String username) throws Exception {
        MvcResult result = mockMvc.perform(post("/api/auth/login")
                        .header("X-Trace-Id", "trace-handoff-test")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("""
                                {
                                  "username": "%s",
                                  "password": "123456"
                                }
                                """.formatted(username)))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.status().isOk())
                .andReturn();
        return objectMapper.readTree(result.getResponse().getContentAsString())
                .path("data").path("token").asText();
    }

    private Map<String, Object> internalBody(String conversationId) {
        return Map.of(
                "conversation_id", conversationId,
                "user_id", "U1001",
                "tenant_id", "default",
                "reason", "检测到强烈情绪（angry），建议由人工客服继续跟进。",
                "related_order_id", "202501010001",
                "emotion", "angry");
    }

    @Test
    void internalCreateHandoffCreatesQueueRecord() throws Exception {
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-1"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        // 幂等：同 conversation 重复写入不产生第二条 active
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-1"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        Integer count = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ai_human_handoffs WHERE conversation_id = 'conv-1'", Integer.class);
        org.junit.jupiter.api.Assertions.assertEquals(1, count);
    }

    @Test
    void internalCreateHandoffRequiresAuth() throws Exception {
        mockMvc.perform(post("/internal/ai-human-handoffs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-2"))))
                .andExpect(status().is4xxClientError());
    }

    @Test
    void claimAndCloseTransitions() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-3"))))
                .andExpect(status().isOk());

        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-3'", Long.class);

        mockMvc.perform(post("/api/human-handoffs/" + id + "/claim")
                        .header("Authorization", "Bearer " + agentToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.status").value("in_progress"));

        mockMvc.perform(post("/api/human-handoffs/" + id + "/close")
                        .header("Authorization", "Bearer " + agentToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("note", "已电话联系客户"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.status").value("closed"));
    }

    @Test
    void claimRejectsNonPending() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-4"))))
                .andExpect(status().isOk());

        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-4'", Long.class);

        mockMvc.perform(post("/api/human-handoffs/" + id + "/close")
                        .header("Authorization", "Bearer " + agentToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("note", "直接关闭"))))
                .andExpect(status().isConflict());
    }

    @Test
    void listByStatusReturnsRecords() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-5"))))
                .andExpect(status().isOk());

        mockMvc.perform(get("/api/human-handoffs")
                        .header("Authorization", "Bearer " + agentToken)
                        .param("status", "pending"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data[0].conversation_id").value("conv-5"))
                .andExpect(jsonPath("$.data[0].emotion").value("angry"));
    }

    @Test
    void transferUpdatesAgentAndKeepsInProgress() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-transfer-1"))))
                .andExpect(status().isOk());
        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-transfer-1'", Long.class);
        mockMvc.perform(post("/api/human-handoffs/" + id + "/claim")
                        .header("Authorization", "Bearer " + agentToken))
                .andExpect(status().isOk());

        mockMvc.perform(post("/api/human-handoffs/" + id + "/transfer")
                        .header("Authorization", "Bearer " + agentToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("target_agent", "A1002"))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.assigned_agent").value("A1002"))
                .andExpect(jsonPath("$.data.status").value("in_progress"))
                .andExpect(jsonPath("$.data.note").value(org.hamcrest.Matchers.containsString("转交")));
    }

    @Test
    void transferRejectsNonInProgress() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-transfer-2"))))
                .andExpect(status().isOk());
        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-transfer-2'", Long.class);
        jdbcTemplate.update("UPDATE ai_human_handoffs SET status = 'closed' WHERE id = ?", id);

        mockMvc.perform(post("/api/human-handoffs/" + id + "/transfer")
                        .header("Authorization", "Bearer " + agentToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("target_agent", "A1002"))))
                .andExpect(status().isConflict());
    }

    @Test
    void transferRejectsSameAgent() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-transfer-3"))))
                .andExpect(status().isOk());
        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-transfer-3'", Long.class);
        mockMvc.perform(post("/api/human-handoffs/" + id + "/claim")
                        .header("Authorization", "Bearer " + agentToken))
                .andExpect(status().isOk());
        String assigned = jdbcTemplate.queryForObject(
                "SELECT assigned_agent FROM ai_human_handoffs WHERE id = ?", String.class, id);

        mockMvc.perform(post("/api/human-handoffs/" + id + "/transfer")
                        .header("Authorization", "Bearer " + agentToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("target_agent", assigned))))
                .andExpect(status().isBadRequest());
    }

    @Test
    void transferRequiresAssignedAgentOrSupervisor() throws Exception {
        String agentToken = loginAndExtractToken("agent");
        String otherToken = loginAndExtractToken("customer");
        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-human-handoffs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(internalBody("conv-transfer-4"))))
                .andExpect(status().isOk());
        Long id = jdbcTemplate.queryForObject(
                "SELECT id FROM ai_human_handoffs WHERE conversation_id = 'conv-transfer-4'", Long.class);
        mockMvc.perform(post("/api/human-handoffs/" + id + "/claim")
                        .header("Authorization", "Bearer " + agentToken))
                .andExpect(status().isOk());
        jdbcTemplate.update("UPDATE ai_human_handoffs SET assigned_agent = 'A9999' WHERE id = ?", id);

        mockMvc.perform(post("/api/human-handoffs/" + id + "/transfer")
                        .header("Authorization", "Bearer " + otherToken)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("target_agent", "A1002"))))
                .andExpect(status().isForbidden());
    }
}
