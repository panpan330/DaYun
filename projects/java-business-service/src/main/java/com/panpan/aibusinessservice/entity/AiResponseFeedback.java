package com.panpan.aibusinessservice.entity;

import java.time.Instant;

public class AiResponseFeedback {
    private Long id;
    private String tenantId;
    private String userId;
    private String conversationId;
    private String traceId;
    private String rating;
    private String reason;
    private String agentRoute;
    private int citationCount;
    private boolean humanHandoffSuggested;
    private String userMessageExcerpt;
    private String assistantAnswerExcerpt;
    private String citationSummaryJson;
    private String reviewStatus;
    private String badCaseId;
    private String reviewedByUserId;
    private Instant reviewedAt;
    private String reviewNote;
    private Instant createdAt;
    private Instant updatedAt;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTenantId() { return tenantId; }
    public void setTenantId(String tenantId) { this.tenantId = tenantId; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getConversationId() { return conversationId; }
    public void setConversationId(String conversationId) { this.conversationId = conversationId; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public String getRating() { return rating; }
    public void setRating(String rating) { this.rating = rating; }
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    public String getAgentRoute() { return agentRoute; }
    public void setAgentRoute(String agentRoute) { this.agentRoute = agentRoute; }
    public int getCitationCount() { return citationCount; }
    public void setCitationCount(int citationCount) { this.citationCount = citationCount; }
    public boolean isHumanHandoffSuggested() { return humanHandoffSuggested; }
    public void setHumanHandoffSuggested(boolean humanHandoffSuggested) { this.humanHandoffSuggested = humanHandoffSuggested; }
    public String getUserMessageExcerpt() { return userMessageExcerpt; }
    public void setUserMessageExcerpt(String userMessageExcerpt) { this.userMessageExcerpt = userMessageExcerpt; }
    public String getAssistantAnswerExcerpt() { return assistantAnswerExcerpt; }
    public void setAssistantAnswerExcerpt(String assistantAnswerExcerpt) { this.assistantAnswerExcerpt = assistantAnswerExcerpt; }
    public String getCitationSummaryJson() { return citationSummaryJson; }
    public void setCitationSummaryJson(String citationSummaryJson) { this.citationSummaryJson = citationSummaryJson; }
    public String getReviewStatus() { return reviewStatus; }
    public void setReviewStatus(String reviewStatus) { this.reviewStatus = reviewStatus; }
    public String getBadCaseId() { return badCaseId; }
    public void setBadCaseId(String badCaseId) { this.badCaseId = badCaseId; }
    public String getReviewedByUserId() { return reviewedByUserId; }
    public void setReviewedByUserId(String reviewedByUserId) { this.reviewedByUserId = reviewedByUserId; }
    public Instant getReviewedAt() { return reviewedAt; }
    public void setReviewedAt(Instant reviewedAt) { this.reviewedAt = reviewedAt; }
    public String getReviewNote() { return reviewNote; }
    public void setReviewNote(String reviewNote) { this.reviewNote = reviewNote; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}
