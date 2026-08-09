package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record HandoffTransferCommand(
        @NotBlank @Size(max = 64) String targetAgent
) {
}
