package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.AiResponseFeedback;
import java.time.Instant;

public record AiResponseFeedbackReceipt(
        long feedbackId,
        String rating,
        String reason,
        Instant updatedAt
) {
    public static AiResponseFeedbackReceipt from(AiResponseFeedback feedback) {
        return new AiResponseFeedbackReceipt(
                feedback.getId(),
                feedback.getRating(),
                feedback.getReason(),
                feedback.getUpdatedAt()
        );
    }
}
