package com.panpan.aibusinessservice.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.util.List;

public record BatchWriteAiMessagesCommand(
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String conversationId,
        @NotEmpty @Size(max = 100) List<@Valid AiMessagePayload> messages
) {}
