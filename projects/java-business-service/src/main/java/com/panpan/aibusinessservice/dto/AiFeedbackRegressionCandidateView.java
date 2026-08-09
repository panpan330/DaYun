package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.AiResponseFeedback;
import java.time.Instant;

public record AiFeedbackRegressionCandidateView(
        long feedbackId,
        String conversationId,
        String traceId,
        String reason,
        String agentRoute,
        int citationCount,
        boolean humanHandoffSuggested,
        String reviewStatus,
        String badCaseId,
        Instant createdAt
) {
    public static AiFeedbackRegressionCandidateView from(AiResponseFeedback feedback) {
        return new AiFeedbackRegressionCandidateView(
                feedback.getId(), feedback.getConversationId(), feedback.getTraceId(), feedback.getReason(),
                feedback.getAgentRoute(), feedback.getCitationCount(), feedback.isHumanHandoffSuggested(),
                feedback.getReviewStatus(), feedback.getBadCaseId(),
                feedback.getCreatedAt()
        );
    }
}
