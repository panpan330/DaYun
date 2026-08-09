package com.panpan.aibusinessservice.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record UpdateTicketStatusRequest(
        @JsonProperty("target_status")
        @NotBlank
        @Pattern(regexp = "^(in_progress|waiting_user|closed)$")
        String targetStatus,

        @Size(max = 500)
        String note
) {
}
