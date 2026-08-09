package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record CreateAiResponseFeedbackCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String conversationId,
        @NotBlank @Size(max = 128) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String traceId,
        @NotBlank @Pattern(regexp = "^(helpful|unhelpful)$") String rating,
        @Size(max = 64) @Pattern(regexp = "^(answer_incorrect|intent_misunderstood|citation_irrelevant|should_handoff|ticket_flow_incorrect|other)?$") String reason,
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String agentRoute,
        @NotNull @Max(100) Integer citationCount,
        @NotNull Boolean humanHandoffSuggested,
        @Size(max = 1000) String userMessageExcerpt,
        @Size(max = 2000) String assistantAnswerExcerpt,
        @Size(max = 4000) String citationSummaryJson
) {}
