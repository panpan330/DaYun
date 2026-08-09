package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record AssignTicketRequest(
        @NotBlank
        @Pattern(regexp = "^[A-Za-z0-9._:-]{1,64}$")
        String assigneeUserId,

        @Size(max = 500)
        String note
) {
}
