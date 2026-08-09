package com.panpan.aibusinessservice.dto;

import jakarta.validation.constraints.NotBlank;

public record LoginRequest(
        @NotBlank String username,
        @NotBlank String password,
        String tenantId
) {
    public String normalizedTenantId() {
        if (tenantId == null || tenantId.isBlank()) {
            return "default";
        }
        return tenantId.trim();
    }
}
