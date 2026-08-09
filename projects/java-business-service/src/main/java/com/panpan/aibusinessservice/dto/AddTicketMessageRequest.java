package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record AddTicketMessageRequest(
        @NotBlank
        @Pattern(regexp = "^(public|internal)$")
        String visibility,

        @NotBlank
        @Size(max = 2000)
        String content
) {
}
