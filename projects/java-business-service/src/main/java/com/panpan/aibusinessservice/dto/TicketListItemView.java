package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.Ticket;
import java.time.Instant;

public record TicketListItemView(
        String ticketId,
        String requesterUserId,
        String ticketStatus,
        String title,
        String category,
        String priority,
        String relatedOrderId,
        String assigneeUserId,
        String assigneeDisplayName,
        String source,
        Instant createdAt,
        Instant updatedAt
) {
    public static TicketListItemView from(Ticket ticket) {
        return new TicketListItemView(
                ticket.getTicketId(),
                ticket.getRequesterUserId(),
                ticket.getTicketStatus(),
                ticket.getTitle(),
                ticket.getCategory(),
                ticket.getPriority(),
                ticket.getRelatedOrderId(),
                ticket.getAssigneeUserId(),
                ticket.getAssigneeDisplayName(),
                ticket.getSource(),
                ticket.getCreatedAt(),
                ticket.getUpdatedAt()
        );
    }
}
