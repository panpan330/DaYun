package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import java.time.Instant;

public record SaveRagEvalRunCommand(
        @NotBlank @Size(max = 64) @Pattern(regexp = "^[A-Za-z0-9_-]+$") String runId,
        @NotBlank @Size(max = 16) @Pattern(regexp = "^(keyword|vector)$") String retriever,
        @NotNull Instant startedAt,
        @NotNull Instant completedAt,
        @Min(1) @Max(100) int topK,
        @DecimalMin("0.0") @DecimalMax("1.0") double hitRate,
        @DecimalMin("0.0") @DecimalMax("1.0") double recall,
        @DecimalMin("0.0") @DecimalMax("1.0") double precision,
        @DecimalMin("0.0") @DecimalMax("1.0") double mrr,
        @Min(1) int caseCount,
        @NotBlank String detailsJson
) {}
