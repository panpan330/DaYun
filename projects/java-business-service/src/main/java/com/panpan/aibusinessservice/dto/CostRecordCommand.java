package com.panpan.aibusinessservice.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;

public record CostRecordCommand(
        @Valid @NotNull List<CostRecordItem> records
) {

    public record CostRecordItem(
            @NotBlank @Size(max = 64) String model,
            @NotBlank @Size(max = 64) String intent,
            @Min(1) int callCount,
            @Min(0) long inputTokens,
            @Min(0) long outputTokens,
            @Min(0) long totalTokens,
            @NotNull @DecimalMin("0.0000") BigDecimal estimatedCost,
            @NotNull Instant windowStart,
            @NotNull Instant windowEnd
    ) {
    }
}
