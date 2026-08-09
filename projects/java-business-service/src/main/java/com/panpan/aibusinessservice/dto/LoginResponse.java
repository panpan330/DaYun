package com.panpan.aibusinessservice.dto;

public record LoginResponse(
        String token,
        CurrentUserView user
) {
}
