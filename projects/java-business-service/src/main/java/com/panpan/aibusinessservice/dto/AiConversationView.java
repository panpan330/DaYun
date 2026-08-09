package com.panpan.aibusinessservice.dto;

import java.time.Instant;

public record AiConversationView(
        String conversationId,
        String userId,
        String title,
        String conversationStatus,
        Instant updatedAt
) {}
