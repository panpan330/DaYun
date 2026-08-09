package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record InternalCreateHandoffCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String conversationId,
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String userId,
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String tenantId,
        @NotBlank @Size(max = 300) String reason,
        @Size(max = 64) String relatedOrderId,
        @Size(max = 32) String emotion
) {
}
