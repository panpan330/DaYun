package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.TicketEvent;
import java.time.Instant;

public record TicketEventView(
        String eventId,
        String eventType,
        String eventPayload,
        String operatorType,
        String operatorId,
        String traceId,
        Instant createdAt
) {
    public static TicketEventView from(TicketEvent event) {
        return new TicketEventView(
                event.getEventId(),
                event.getEventType(),
                event.getEventPayload(),
                event.getOperatorType(),
                event.getOperatorId(),
                event.getTraceId(),
                event.getCreatedAt()
        );
    }
}
