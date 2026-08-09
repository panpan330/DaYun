package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

@SpringBootTest
@AutoConfigureMockMvc
class InternalApiContractTest {
    private static final Path CONTRACT_PATH = Path.of(
            "..",
            "..",
            "contracts",
            "java-business-service",
            "internal-api-contract-cases.json"
    );

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void queryOrderSuccessMatchesSharedContract() throws Exception {
        JsonNode item = contractCase("query_order_success");

        mockMvc.perform(withContractHeaders(get(item.path("path").asText())))
                .andExpect(status().is(item.path("expected").path("status").asInt()))
                .andExpect(header().string(TraceHeaders.TRACE_ID, contractTraceId()))
                .andExpect(jsonPath("$.success").value(item.path("expected").path("success").asBoolean()))
                .andExpect(jsonPath("$.code").value(item.path("expected").path("code").asText()))
                .andExpect(jsonPath("$.trace_id").value(contractTraceId()))
                .andExpect(jsonPath("$.data.order_id").exists())
                .andExpect(jsonPath("$.data.order_status").exists())
                .andExpect(jsonPath("$.data.payment_status").exists())
                .andExpect(jsonPath("$.data.logistics_message").exists())
                .andExpect(jsonPath("$.data.latest_event").exists())
                .andExpect(jsonPath("$.data.can_create_ticket").exists())
                .andExpect(jsonPath("$.data.user_visible_summary").exists());
    }

    @Test
    void queryOrderAccessDeniedMatchesSharedContract() throws Exception {
        JsonNode item = contractCase("query_order_access_denied");

        mockMvc.perform(withContractHeaders(get(item.path("path").asText())))
                .andExpect(status().is(item.path("expected").path("status").asInt()))
                .andExpect(header().string(TraceHeaders.TRACE_ID, contractTraceId()))
                .andExpect(jsonPath("$.success").value(item.path("expected").path("success").asBoolean()))
                .andExpect(jsonPath("$.code").value(item.path("expected").path("code").asText()))
                .andExpect(jsonPath("$.trace_id").value(contractTraceId()));
    }

    @Test
    void createTicketSuccessMatchesSharedContract() throws Exception {
        JsonNode item = contractCase("create_ticket_success");

        mockMvc.perform(withContractHeaders(post(item.path("path").asText()))
                        .header(TraceHeaders.IDEMPOTENCY_KEY, item.path("idempotency_key").asText())
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(item.path("body"))))
                .andExpect(status().is(item.path("expected").path("status").asInt()))
                .andExpect(header().string(TraceHeaders.TRACE_ID, contractTraceId()))
                .andExpect(jsonPath("$.success").value(item.path("expected").path("success").asBoolean()))
                .andExpect(jsonPath("$.code").value(item.path("expected").path("code").asText()))
                .andExpect(jsonPath("$.trace_id").value(contractTraceId()))
                .andExpect(jsonPath("$.data.ticket_id").exists())
                .andExpect(jsonPath("$.data.ticket_status").exists())
                .andExpect(jsonPath("$.data.title").exists())
                .andExpect(jsonPath("$.data.category").exists())
                .andExpect(jsonPath("$.data.priority").exists())
                .andExpect(jsonPath("$.data.related_order_id").exists())
                .andExpect(jsonPath("$.data.created_at").exists())
                .andExpect(jsonPath("$.data.user_visible_summary").exists());
    }

    @Test
    void createTicketMissingIdempotencyKeyMatchesSharedContract() throws Exception {
        JsonNode item = contractCase("create_ticket_missing_idempotency_key");

        mockMvc.perform(withContractHeaders(post(item.path("path").asText()))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(item.path("body"))))
                .andExpect(status().is(item.path("expected").path("status").asInt()))
                .andExpect(header().string(TraceHeaders.TRACE_ID, contractTraceId()))
                .andExpect(jsonPath("$.success").value(item.path("expected").path("success").asBoolean()))
                .andExpect(jsonPath("$.code").value(item.path("expected").path("code").asText()))
                .andExpect(jsonPath("$.trace_id").value(contractTraceId()));
    }

    private MockHttpServletRequestBuilder withContractHeaders(MockHttpServletRequestBuilder request) throws Exception {
        JsonNode headers = contract().path("common_headers");
        return request
                .header(TraceHeaders.TRACE_ID, headers.path(TraceHeaders.TRACE_ID).asText())
                .header(TraceHeaders.CALLER, headers.path(TraceHeaders.CALLER).asText())
                .header(TraceHeaders.USER_ID, headers.path(TraceHeaders.USER_ID).asText())
                .header(TraceHeaders.TENANT_ID, headers.path(TraceHeaders.TENANT_ID).asText())
                .header(TraceHeaders.INTERNAL_TOKEN, headers.path(TraceHeaders.INTERNAL_TOKEN).asText());
    }

    private String contractTraceId() throws Exception {
        return contract().path("common_headers").path(TraceHeaders.TRACE_ID).asText();
    }

    private JsonNode contractCase(String id) throws Exception {
        for (JsonNode item : contract().path("cases")) {
            if (id.equals(item.path("id").asText())) {
                return item;
            }
        }
        throw new IllegalArgumentException("Missing contract case: " + id);
    }

    private JsonNode contract() throws Exception {
        return objectMapper.readTree(CONTRACT_PATH.toFile());
    }
}
