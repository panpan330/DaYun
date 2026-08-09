package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record AiMessagePayload(
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String messageId,
        @NotBlank @Size(max = 32) @Pattern(regexp = "^(user|assistant|human_agent)$") String senderType,
        @NotBlank @Size(max = 20000) String content,
        @NotBlank @Size(max = 128) @Pattern(regexp = "^[A-Za-z0-9._:-]+$") String traceId
) {}
