package com.panpan.aibusinessservice;

import static com.panpan.aibusinessservice.InternalApiTestSupport.TRACE_ID;
import static com.panpan.aibusinessservice.InternalApiTestSupport.withInternalHeaders;
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
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

@SpringBootTest
@AutoConfigureMockMvc
class InternalTicketControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Test
    void createTicketReturnsToolFacingView() throws Exception {
        MvcResult result = mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .header(TraceHeaders.IDEMPOTENCY_KEY, "ticket-stage7-create-001")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson())
                )
                .andExpect(status().isCreated())
                .andExpect(header().string(TraceHeaders.TRACE_ID, TRACE_ID))
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.code").value("OK"))
                .andExpect(jsonPath("$.data.ticket_id").exists())
                .andExpect(jsonPath("$.data.ticket_status").value("created"))
                .andExpect(jsonPath("$.data.related_order_id").value("A1001"))
                .andExpect(jsonPath("$.trace_id").value(TRACE_ID))
                .andReturn();

        String body = result.getResponse().getContentAsString();
        JsonNode json = objectMapper.readTree(body);
        String ticketId = json.path("data").path("ticket_id").asText();

        Integer ticketCount = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM tickets WHERE tenant_id = ? AND ticket_id = ?",
                Integer.class,
                "default",
                ticketId
        );
        Integer eventCount = jdbcTemplate.queryForObject(
                "SELECT COUNT(*) FROM ticket_events WHERE tenant_id = ? AND ticket_id = ? AND event_type = ? AND trace_id = ?",
                Integer.class,
                "default",
                ticketId,
                "created",
                TRACE_ID
        );

        org.assertj.core.api.Assertions.assertThat(ticketCount).isEqualTo(1);
        org.assertj.core.api.Assertions.assertThat(eventCount).isEqualTo(1);
    }

    @Test
    void createTicketRequiresIdempotencyKey() throws Exception {
        mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson())
                )
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("IDEMPOTENCY_KEY_REQUIRED"));
    }

    @Test
    void createTicketIsIdempotentForSameKeyAndSamePayload() throws Exception {
        String idempotencyKey = "ticket-stage7-create-002";

        MvcResult firstResult = mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .header(TraceHeaders.IDEMPOTENCY_KEY, idempotencyKey)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson())
                )
                .andExpect(status().isCreated())
                .andReturn();

        String firstBody = firstResult.getResponse().getContentAsString();
        JsonNode firstJson = objectMapper.readTree(firstBody);
        String ticketId = firstJson.path("data").path("ticket_id").asText();

        mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .header(TraceHeaders.IDEMPOTENCY_KEY, idempotencyKey)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson())
                )
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.data.ticket_id").value(ticketId));
    }

    @Test
    void createTicketRejectsIdempotencyKeyConflict() throws Exception {
        String idempotencyKey = "ticket-stage7-create-003";

        mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .header(TraceHeaders.IDEMPOTENCY_KEY, idempotencyKey)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson())
                )
                .andExpect(status().isCreated());

        mockMvc.perform(
                        withInternalHeaders(post("/internal/tickets"))
                                .header(TraceHeaders.IDEMPOTENCY_KEY, idempotencyKey)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content(validCreateTicketJson().replace("物流太慢", "订单投诉"))
                )
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("IDEMPOTENCY_KEY_CONFLICT"));
    }

    private String validCreateTicketJson() {
        return """
                {
                  "title": "物流太慢",
                  "description": "用户反馈 A1001 订单物流长时间未更新，希望客服跟进。",
                  "category": "logistics",
                  "priority": "normal",
                  "related_order_id": "A1001",
                  "source": "ai_agent",
                  "confirmation_id": "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11"
                }
                """;
    }
}
