package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.Ticket;
import java.time.Instant;
import java.util.List;

public record TicketDetailView(
        String ticketId,
        String requesterUserId,
        String ticketStatus,
        String title,
        String description,
        String category,
        String priority,
        String relatedOrderId,
        String assigneeUserId,
        String assigneeDisplayName,
        String source,
        String createdTraceId,
        Instant createdAt,
        Instant updatedAt,
        List<TicketEventView> events,
        List<TicketMessageView> messages
) {
    public static TicketDetailView from(
            Ticket ticket,
            List<TicketEventView> events,
            List<TicketMessageView> messages
    ) {
        return new TicketDetailView(
                ticket.getTicketId(),
                ticket.getRequesterUserId(),
                ticket.getTicketStatus(),
                ticket.getTitle(),
                ticket.getDescription(),
                ticket.getCategory(),
                ticket.getPriority(),
                ticket.getRelatedOrderId(),
                ticket.getAssigneeUserId(),
                ticket.getAssigneeDisplayName(),
                ticket.getSource(),
                ticket.getCreatedTraceId(),
                ticket.getCreatedAt(),
                ticket.getUpdatedAt(),
                events,
                messages
        );
    }
}
