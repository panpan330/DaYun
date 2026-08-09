package com.panpan.aibusinessservice.entity;

import java.time.Instant;

public class Ticket {
    private String ticketId;
    private String requesterUserId;
    private String tenantId;
    private String ticketStatus;
    private String title;
    private String description;
    private String category;
    private String priority;
    private String relatedOrderId;
    private String assigneeUserId;
    private String assigneeDisplayName;
    private String source;
    private String confirmationId;
    private String idempotencyKey;
    private String requestFingerprint;
    private String createdTraceId;
    private Instant createdAt;
    private Instant updatedAt;

    public Ticket() {
    }

    public Ticket(
            String ticketId,
            String requesterUserId,
            String tenantId,
            String ticketStatus,
            String title,
            String description,
            String category,
            String priority,
            String relatedOrderId,
            String source,
            String confirmationId,
            String idempotencyKey,
            String requestFingerprint,
            String createdTraceId,
            Instant createdAt
    ) {
        this.ticketId = ticketId;
        this.requesterUserId = requesterUserId;
        this.tenantId = tenantId;
        this.ticketStatus = ticketStatus;
        this.title = title;
        this.description = description;
        this.category = category;
        this.priority = priority;
        this.relatedOrderId = relatedOrderId;
        this.source = source;
        this.confirmationId = confirmationId;
        this.idempotencyKey = idempotencyKey;
        this.requestFingerprint = requestFingerprint;
        this.createdTraceId = createdTraceId;
        this.createdAt = createdAt;
    }

    public String getTicketId() {
        return ticketId;
    }

    public void setTicketId(String ticketId) {
        this.ticketId = ticketId;
    }

    public String getRequesterUserId() {
        return requesterUserId;
    }

    public void setRequesterUserId(String requesterUserId) {
        this.requesterUserId = requesterUserId;
    }

    public String getTenantId() {
        return tenantId;
    }

    public void setTenantId(String tenantId) {
        this.tenantId = tenantId;
    }

    public String getTicketStatus() {
        return ticketStatus;
    }

    public void setTicketStatus(String ticketStatus) {
        this.ticketStatus = ticketStatus;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public String getPriority() {
        return priority;
    }

    public void setPriority(String priority) {
        this.priority = priority;
    }

    public String getRelatedOrderId() {
        return relatedOrderId;
    }

    public void setRelatedOrderId(String relatedOrderId) {
        this.relatedOrderId = relatedOrderId;
    }

    public String getAssigneeUserId() {
        return assigneeUserId;
    }

    public void setAssigneeUserId(String assigneeUserId) {
        this.assigneeUserId = assigneeUserId;
    }

    public String getAssigneeDisplayName() {
        return assigneeDisplayName;
    }

    public void setAssigneeDisplayName(String assigneeDisplayName) {
        this.assigneeDisplayName = assigneeDisplayName;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getConfirmationId() {
        return confirmationId;
    }

    public void setConfirmationId(String confirmationId) {
        this.confirmationId = confirmationId;
    }

    public String getIdempotencyKey() {
        return idempotencyKey;
    }

    public void setIdempotencyKey(String idempotencyKey) {
        this.idempotencyKey = idempotencyKey;
    }

    public String getRequestFingerprint() {
        return requestFingerprint;
    }

    public void setRequestFingerprint(String requestFingerprint) {
        this.requestFingerprint = requestFingerprint;
    }

    public String getCreatedTraceId() {
        return createdTraceId;
    }

    public void setCreatedTraceId(String createdTraceId) {
        this.createdTraceId = createdTraceId;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }
}
