package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.TicketMessage;
import java.time.Instant;

public record TicketMessageView(
        String messageId,
        String visibility,
        String content,
        String authorType,
        String authorUserId,
        String authorDisplayName,
        Instant createdAt
) {
    public static TicketMessageView from(TicketMessage message) {
        return new TicketMessageView(
                message.getMessageId(),
                message.getVisibility(),
                message.getContent(),
                message.getAuthorType(),
                message.getAuthorUserId(),
                message.getAuthorDisplayName(),
                message.getCreatedAt()
        );
    }
}
