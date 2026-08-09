package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.TicketCategory;
import com.panpan.aibusinessservice.entity.TicketPriority;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record CreateTicketCommand(
        @NotBlank @Size(max = 200) String title,
        @NotBlank @Size(max = 1000) String description,
        @NotNull TicketCategory category,
        @NotNull TicketPriority priority,
        @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String relatedOrderId,
        @NotBlank @Pattern(regexp = "^ai_agent$") String source,
        @NotBlank @Pattern(regexp = "^[a-f0-9]{32}$") String confirmationId
) {
}
