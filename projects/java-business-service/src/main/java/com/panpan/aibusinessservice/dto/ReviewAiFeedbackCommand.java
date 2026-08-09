package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record ReviewAiFeedbackCommand(
        @NotBlank @Pattern(regexp = "^(triaged|closed)$") String reviewStatus,
        @Size(max = 1000) String reviewNote
) {}
