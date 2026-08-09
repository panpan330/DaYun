package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
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
class InternalAiCostRecordControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_cost_records");
    }

    @Test
    void batchUpsertAndOverview() throws Exception {
        Instant now = Instant.now();
        Map<String, Object> item = Map.of(
                "model", "gpt-4o",
                "intent", "order_query",
                "call_count", 2,
                "input_tokens", 300,
                "output_tokens", 150,
                "total_tokens", 450,
                "estimated_cost", 0.03,
                "window_start", now.toString(),
                "window_end", now.toString());

        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(post("/internal/ai-cost-records"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("records", List.of(item)))))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        mockMvc.perform(InternalApiTestSupport.withInternalHeaders(get("/internal/ai-cost-records/overview")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.by_model").isArray())
                .andExpect(jsonPath("$.data.by_intent").isArray())
                .andExpect(jsonPath("$.data.totals.call_count").value(2));
    }

    @Test
    void batchUpsertRequiresInternalAuth() throws Exception {
        Map<String, Object> item = Map.of(
                "model", "gpt-4o",
                "intent", "general",
                "call_count", 1,
                "input_tokens", 1,
                "output_tokens", 1,
                "total_tokens", 2,
                "estimated_cost", 0.001,
                "windowStart", Instant.now().toString(),
                "windowEnd", Instant.now().toString());

        mockMvc.perform(post("/internal/ai-cost-records")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("records", List.of(item)))))
                .andExpect(status().is4xxClientError());
    }
}
