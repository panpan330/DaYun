package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.Ticket;
import java.time.Instant;

public record TicketToolView(
        String ticketId,
        String ticketStatus,
        String title,
        String category,
        String priority,
        String relatedOrderId,
        Instant createdAt,
        String userVisibleSummary
) {
    public static TicketToolView from(Ticket ticket) {
        return new TicketToolView(
                ticket.getTicketId(),
                ticket.getTicketStatus(),
                ticket.getTitle(),
                ticket.getCategory(),
                ticket.getPriority(),
                ticket.getRelatedOrderId(),
                ticket.getCreatedAt(),
                "工单已创建，客服会继续跟进。"
        );
    }
}
