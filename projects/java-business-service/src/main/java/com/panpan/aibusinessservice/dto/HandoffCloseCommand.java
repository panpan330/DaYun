package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.Size;

public record HandoffCloseCommand(
        @Size(max = 500) String note
) {
}
