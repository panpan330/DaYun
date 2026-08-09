package com.panpan.aibusinessservice.dto;

import java.time.Instant;

public record HumanHandoffView(
        Long id,
        String conversationId,
        String userId,
        String reason,
        String relatedOrderId,
        String emotion,
        String status,
        String assignedAgent,
        String note,
        Instant createdAt,
        Instant resolvedAt
) {
}
