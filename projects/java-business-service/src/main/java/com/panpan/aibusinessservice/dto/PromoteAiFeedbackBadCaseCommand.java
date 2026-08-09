package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record PromoteAiFeedbackBadCaseCommand(
        @NotBlank @Size(max = 160) @Pattern(regexp = "^[a-z0-9_]+$") String badCaseId
) {}
