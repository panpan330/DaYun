package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record UpsertAiConversationCommand(
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String conversationId,
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String userId,
        @NotBlank @Size(max = 200) String title,
        @NotBlank @Size(max = 32) @Pattern(regexp = "^(active|completed)$") String conversationStatus
) {}
