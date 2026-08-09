package com.panpan.aibusinessservice.dto;

import java.time.Instant;

public record AiMessageView(
        String messageId,
        String conversationId,
        String senderType,
        String content,
        String traceId,
        Instant createdAt
) {}
