package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record ReopenTicketRequest(
        @NotBlank
        @Size(max = 2000)
        String content
) {
}
