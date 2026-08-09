package com.panpan.aibusinessservice;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
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
import org.springframework.transaction.annotation.Transactional;

@SpringBootTest
@AutoConfigureMockMvc
class PublicOrderTicketControllerTest {
    private static final String TRACE_ID = "trace-stage11-list-test";

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Test
    void customerListsOnlyOwnOrders() throws Exception {
        String token = loginAndExtractToken("customer");

        MvcResult result = mockMvc.perform(
                        get("/api/orders")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.length()").value(2))
                .andReturn();

        JsonNode data = objectMapper.readTree(result.getResponse().getContentAsString()).path("data");
        for (JsonNode order : data) {
            assertThat(order.path("owner_user_id").asText()).isEqualTo("U1001");
        }
    }

    @Test
    void agentListsTenantOrders() throws Exception {
        String token = loginAndExtractToken("agent");

        mockMvc.perform(
                        get("/api/orders")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.length()").value(3));
    }

    @Test
    void customerListsOnlyOwnTickets() throws Exception {
        String token = loginAndExtractToken("customer");

        MvcResult result = mockMvc.perform(
                        get("/api/tickets")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andReturn();

        JsonNode data = objectMapper.readTree(result.getResponse().getContentAsString()).path("data");
        assertThat(data).isNotEmpty();
        for (JsonNode ticket : data) {
            assertThat(ticket.path("requester_user_id").asText()).isEqualTo("U1001");
        }
    }

    @Test
    void publicListsRejectMissingAuthorization() throws Exception {
        mockMvc.perform(
                        get("/api/orders")
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                )
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("AUTH_REQUIRED"));
    }

    @Test
    void agentGetsTicketDetailWithEvents() throws Exception {
        String ticketId = "T-STAGE11-DETAIL";
        seedTicket(ticketId, "U1001", "created");
        seedEvent(ticketId, "E-STAGE11-DETAIL", "created", """
                {"title":"seeded ticket"}
                """);
        String token = loginAndExtractToken("agent");

        mockMvc.perform(
                        get("/api/tickets/{ticketId}", ticketId)
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_id").value(ticketId))
                .andExpect(jsonPath("$.data.events[0].event_type").value("created"));
    }

    @Test
    void agentUpdatesTicketStatusAndWritesEvent() throws Exception {
        String ticketId = "T-STAGE11-STATUS";
        seedTicket(ticketId, "U1001", "created");
        String token = loginAndExtractToken("agent");

        mockMvc.perform(
                        patch("/api/tickets/{ticketId}/status", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-status")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "target_status": "in_progress",
                                          "note": "manual follow up"
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_id").value(ticketId))
                .andExpect(jsonPath("$.data.ticket_status").value("in_progress"));

        String status = jdbcTemplate.queryForObject(
                "SELECT ticket_status FROM tickets WHERE tenant_id = ? AND ticket_id = ?",
                String.class,
                "default",
                ticketId
        );
        Integer eventCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_events
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND event_type = 'status_changed'
                        """,
                Integer.class,
                "default",
                ticketId
        );

        assertThat(status).isEqualTo("in_progress");
        assertThat(eventCount).isEqualTo(1);
    }

    @Test
    void customerCannotUpdateTicketStatus() throws Exception {
        String ticketId = "T-STAGE11-CUSTOMER-DENIED";
        seedTicket(ticketId, "U1001", "created");
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        patch("/api/tickets/{ticketId}/status", ticketId)
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "target_status": "in_progress"
                                        }
                                        """)
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("TICKET_ACCESS_DENIED"));
    }

    @Test
    void agentClaimsUnassignedTicketAndWritesEvent() throws Exception {
        String ticketId = "T-STAGE11-CLAIM";
        seedTicket(ticketId, "U1001", "created");
        String token = loginAndExtractToken("agent");

        mockMvc.perform(
                        patch("/api/tickets/{ticketId}/assignment/claim", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-claim")
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_id").value(ticketId))
                .andExpect(jsonPath("$.data.assignee_user_id").value("A1001"))
                .andExpect(jsonPath("$.data.assignee_display_name").value("Demo Agent"));

        Integer assignmentCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_assignments
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND assignee_user_id = 'A1001'
                        """,
                Integer.class,
                "default",
                ticketId
        );
        Integer eventCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_events
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND event_type = 'ticket_claimed'
                        """,
                Integer.class,
                "default",
                ticketId
        );

        assertThat(assignmentCount).isEqualTo(1);
        assertThat(eventCount).isEqualTo(1);
    }

    @Test
    void supervisorAssignsTicketToAgent() throws Exception {
        String ticketId = "T-STAGE11-ASSIGN";
        seedTicket(ticketId, "U1001", "created");
        String token = loginAndExtractToken("supervisor");

        mockMvc.perform(
                        patch("/api/tickets/{ticketId}/assignment", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-assign")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "assignee_user_id": "A1001",
                                          "note": "assign to first-line agent"
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_id").value(ticketId))
                .andExpect(jsonPath("$.data.assignee_user_id").value("A1001"));

        String assigneeUserId = jdbcTemplate.queryForObject(
                "SELECT assignee_user_id FROM ticket_assignments WHERE tenant_id = ? AND ticket_id = ?",
                String.class,
                "default",
                ticketId
        );
        assertThat(assigneeUserId).isEqualTo("A1001");
    }

    @Test
    void customerCannotClaimTicket() throws Exception {
        String ticketId = "T-STAGE11-CLAIM-DENIED";
        seedTicket(ticketId, "U1001", "created");
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        patch("/api/tickets/{ticketId}/assignment/claim", ticketId)
                                .header(TraceHeaders.TRACE_ID, TRACE_ID)
                                .header("Authorization", "Bearer " + token)
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("TICKET_ACCESS_DENIED"));
    }

    @Test
    void agentAddsPublicAndInternalMessagesButCustomerSeesOnlyPublicReply() throws Exception {
        String ticketId = "T-STAGE11-MESSAGE-VISIBILITY";
        seedTicket(ticketId, "U1001", "in_progress");
        String agentToken = loginAndExtractToken("agent");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/messages", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-public-message")
                                .header("Authorization", "Bearer " + agentToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "visibility": "public",
                                          "content": "We have contacted the carrier and will update you tomorrow."
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.messages.length()").value(1))
                .andExpect(jsonPath("$.data.messages[0].visibility").value("public"));

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/messages", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-internal-message")
                                .header("Authorization", "Bearer " + agentToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "visibility": "internal",
                                          "content": "Carrier escalation reference: OPS-20260804-001."
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.messages.length()").value(2));

        String customerToken = loginAndExtractToken("customer");
        MvcResult customerResult = mockMvc.perform(
                        get("/api/tickets/{ticketId}", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-customer-message-read")
                                .header("Authorization", "Bearer " + customerToken)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.messages.length()").value(1))
                .andExpect(jsonPath("$.data.messages[0].visibility").value("public"))
                .andReturn();

        String customerBody = customerResult.getResponse().getContentAsString();
        assertThat(customerBody).contains("We have contacted the carrier")
                .doesNotContain("OPS-20260804-001");
    }

    @Test
    void customerPublicReplyResumesWaitingTicketAndWritesCustomerMessage() throws Exception {
        String ticketId = "T-STAGE11-CUSTOMER-REPLY";
        seedTicket(ticketId, "U1001", "waiting_user");
        String customerToken = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/messages", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-customer-reply")
                                .header("Authorization", "Bearer " + customerToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "visibility": "public",
                                          "content": "I have confirmed the delivery address and attached the requested details."
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_status").value("in_progress"))
                .andExpect(jsonPath("$.data.messages.length()").value(1))
                .andExpect(jsonPath("$.data.messages[0].author_type").value("customer"));

        String ticketStatus = jdbcTemplate.queryForObject(
                "SELECT ticket_status FROM tickets WHERE tenant_id = ? AND ticket_id = ?",
                String.class,
                "default",
                ticketId
        );
        Integer messageCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_messages
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND author_type = 'customer'
                          AND visibility = 'public'
                        """,
                Integer.class,
                "default",
                ticketId
        );
        Integer resumedEventCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_events
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND event_type = 'status_changed'
                          AND operator_type = 'customer'
                        """,
                Integer.class,
                "default",
                ticketId
        );

        assertThat(ticketStatus).isEqualTo("in_progress");
        assertThat(messageCount).isEqualTo(1);
        assertThat(resumedEventCount).isEqualTo(1);
    }

    @Test
    void customerCannotWriteInternalMessageOrReplyToResolvedTicket() throws Exception {
        String internalTicketId = "T-STAGE11-CUSTOMER-INTERNAL-DENIED";
        seedTicket(internalTicketId, "U1001", "in_progress");
        String customerToken = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/messages", internalTicketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-customer-internal-denied")
                                .header("Authorization", "Bearer " + customerToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "visibility": "internal",
                                          "content": "This must not become an internal note."
                                        }
                                        """)
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.code").value("TICKET_MESSAGE_VISIBILITY_INVALID"));

        String resolvedTicketId = "T-STAGE11-CUSTOMER-RESOLVED-DENIED";
        seedTicket(resolvedTicketId, "U1001", "resolved");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/messages", resolvedTicketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-customer-resolved-denied")
                                .header("Authorization", "Bearer " + customerToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "visibility": "public",
                                          "content": "I need to provide more information."
                                        }
                                        """)
                )
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("TICKET_CUSTOMER_REPLY_NOT_ALLOWED"));
    }

    @Test
    void agentResolvesTicketWithPublicMessageAndCustomerCanReopenIt() throws Exception {
        String ticketId = "T-STAGE11-RESOLVE-REOPEN";
        seedTicket(ticketId, "U1001", "in_progress");
        String agentToken = loginAndExtractToken("agent");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/resolution", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-resolve")
                                .header("Authorization", "Bearer " + agentToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "content": "The carrier has confirmed the delivery schedule. Please check for the parcel tomorrow."
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_status").value("resolved"))
                .andExpect(jsonPath("$.data.messages[0].visibility").value("public"))
                .andExpect(jsonPath("$.data.messages[0].author_type").value("staff"));

        String customerToken = loginAndExtractToken("customer");
        mockMvc.perform(
                        post("/api/tickets/{ticketId}/reopen", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-reopen")
                                .header("Authorization", "Bearer " + customerToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "content": "The parcel is still not delivered, so the issue has not been resolved."
                                        }
                                        """)
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.ticket_status").value("in_progress"))
                .andExpect(jsonPath("$.data.messages.length()").value(2))
                .andExpect(jsonPath("$.data.messages[1].author_type").value("customer"));

        String ticketStatus = jdbcTemplate.queryForObject(
                "SELECT ticket_status FROM tickets WHERE tenant_id = ? AND ticket_id = ?",
                String.class,
                "default",
                ticketId
        );
        Integer reopenedEventCount = jdbcTemplate.queryForObject(
                """
                        SELECT COUNT(*)
                        FROM ticket_events
                        WHERE tenant_id = ?
                          AND ticket_id = ?
                          AND event_type = 'ticket_reopened'
                        """,
                Integer.class,
                "default",
                ticketId
        );

        assertThat(ticketStatus).isEqualTo("in_progress");
        assertThat(reopenedEventCount).isEqualTo(1);
    }

    @Test
    void customerCannotReopenTicketThatIsNotResolved() throws Exception {
        String ticketId = "T-STAGE11-REOPEN-DENIED";
        seedTicket(ticketId, "U1001", "in_progress");
        String customerToken = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/tickets/{ticketId}/reopen", ticketId)
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-ticket-reopen-denied")
                                .header("Authorization", "Bearer " + customerToken)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("""
                                        {
                                          "content": "This must not reopen a ticket that is already being processed."
                                        }
                                        """)
                )
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.code").value("TICKET_REOPEN_NOT_ALLOWED"));
    }

    @Test
    @Transactional
    void customerCanRefundOwnUnshippedOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1002/refund")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-refund-own")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"七天无理由退货\"}")
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.order_id").value("A1002"))
                .andExpect(jsonPath("$.data.order_status").value("waiting_shipment"))
                .andExpect(jsonPath("$.data.payment_status").value("refunded"))
                .andExpect(jsonPath("$.data.refund_amount").value(closeTo(159.00, 0.001)))
                .andExpect(jsonPath("$.data.refund_reason").value("七天无理由退货"))
                .andExpect(jsonPath("$.data.latest_event").value("退款成功"));
    }

    @Test
    @Transactional
    void customerCannotRefundShippedOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1001/refund")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-refund-shipped")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"想退款\"}")
                )
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("ORDER_NOT_REFUNDABLE"));
    }

    @Test
    void customerCannotRefundWithoutToken() throws Exception {
        mockMvc.perform(
                        post("/api/orders/A1002/refund")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-refund-no-token")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"七天无理由退货\"}")
                )
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("AUTH_REQUIRED"));
    }

    @Test
    void customerCannotRefundWithoutReason() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1002/refund")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-refund-no-reason")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{}")
                )
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("REFUND_REASON_REQUIRED"));
    }

    @Test
    @Transactional
    void customerCannotRefundOthersOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A2001/refund")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-refund-other-order")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"想退款\"}")
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("ORDER_ACCESS_DENIED"));
    }

    @Test
    @Transactional
    void customerCanCancelOwnUnshippedOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1002/cancel")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-cancel-own")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"七天无理由取消\"}")
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.order_id").value("A1002"))
                .andExpect(jsonPath("$.data.order_status").value("canceled"))
                .andExpect(jsonPath("$.data.payment_status").value("paid"))
                .andExpect(jsonPath("$.data.cancel_reason").value("七天无理由取消"))
                .andExpect(jsonPath("$.data.latest_event").value("订单已取消"));
    }

    @Test
    @Transactional
    void customerCannotCancelShippedOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1001/cancel")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-cancel-shipped")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"想取消\"}")
                )
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("ORDER_NOT_CANCELABLE"));
    }

    @Test
    void customerCannotCancelWithoutToken() throws Exception {
        mockMvc.perform(
                        post("/api/orders/A1002/cancel")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-cancel-no-token")
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"七天无理由取消\"}")
                )
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("AUTH_REQUIRED"));
    }

    @Test
    void customerCannotCancelWithoutReason() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A1002/cancel")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-cancel-no-reason")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{}")
                )
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("CANCEL_REASON_REQUIRED"));
    }

    @Test
    @Transactional
    void customerCannotCancelOthersOrder() throws Exception {
        String token = loginAndExtractToken("customer");

        mockMvc.perform(
                        post("/api/orders/A2001/cancel")
                                .header(TraceHeaders.TRACE_ID, "trace-stage11-cancel-other-order")
                                .header("Authorization", "Bearer " + token)
                                .contentType(MediaType.APPLICATION_JSON)
                                .content("{\"reason\": \"想取消\"}")
                )
                .andExpect(status().isForbidden())
                .andExpect(jsonPath("$.success").value(false))
                .andExpect(jsonPath("$.code").value("ORDER_ACCESS_DENIED"));
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

    private void seedTicket(String ticketId, String requesterUserId, String ticketStatus) {
        jdbcTemplate.update(
                "DELETE FROM ticket_messages WHERE tenant_id = ? AND ticket_id = ?",
                "default",
                ticketId
        );
        jdbcTemplate.update(
                "DELETE FROM ticket_events WHERE tenant_id = ? AND ticket_id = ?",
                "default",
                ticketId
        );
        jdbcTemplate.update(
                "DELETE FROM ticket_assignments WHERE tenant_id = ? AND ticket_id = ?",
                "default",
                ticketId
        );
        jdbcTemplate.update(
                "DELETE FROM tickets WHERE tenant_id = ? AND ticket_id = ?",
                "default",
                ticketId
        );
        jdbcTemplate.update(
                """
                        INSERT INTO tickets (
                          tenant_id,
                          ticket_id,
                          requester_user_id,
                          related_order_id,
                          title,
                          description,
                          category,
                          priority,
                          ticket_status,
                          source,
                          confirmation_id,
                          idempotency_key,
                          request_fingerprint,
                          created_trace_id,
                          created_at,
                          updated_at
                        ) VALUES (
                          'default',
                          ?,
                          ?,
                          'A1001',
                          'Stage 11 ticket',
                          'Stage 11 ticket workbench test case.',
                          'logistics',
                          'normal',
                          ?,
                          'manual',
                          ?,
                          ?,
                          ?,
                          'trace-stage11-seed',
                          CURRENT_TIMESTAMP(6),
                          CURRENT_TIMESTAMP(6)
                        )
                        """,
                ticketId,
                requesterUserId,
                ticketStatus,
                "confirmation-" + ticketId,
                "idempotency-" + ticketId,
                "fingerprint-" + ticketId
        );
    }

    private void seedEvent(String ticketId, String eventId, String eventType, String payload) {
        jdbcTemplate.update(
                """
                        INSERT INTO ticket_events (
                          tenant_id,
                          event_id,
                          ticket_id,
                          event_type,
                          event_payload,
                          operator_type,
                          operator_id,
                          trace_id,
                          created_at
                        ) VALUES (
                          'default',
                          ?,
                          ?,
                          ?,
                          ?,
                          'system',
                          'test',
                          'trace-stage11-seed',
                          CURRENT_TIMESTAMP(6)
                        )
                        """,
                eventId,
                ticketId,
                eventType,
                payload
        );
    }
}
