package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.AiResponseFeedback;

public record AiResponseFeedbackContextView(
        long feedbackId,
        String conversationId,
        String traceId,
        String reason,
        String agentRoute,
        int citationCount,
        boolean humanHandoffSuggested,
        String userMessageExcerpt,
        String assistantAnswerExcerpt,
        String citationSummaryJson,
        String reviewStatus,
        String badCaseId,
        String reviewNote
) {
    public static AiResponseFeedbackContextView from(AiResponseFeedback feedback) {
        return new AiResponseFeedbackContextView(
                feedback.getId(), feedback.getConversationId(), feedback.getTraceId(), feedback.getReason(),
                feedback.getAgentRoute(), feedback.getCitationCount(), feedback.isHumanHandoffSuggested(),
                feedback.getUserMessageExcerpt(), feedback.getAssistantAnswerExcerpt(), feedback.getCitationSummaryJson(),
                feedback.getReviewStatus(), feedback.getBadCaseId(), feedback.getReviewNote()
        );
    }
}
