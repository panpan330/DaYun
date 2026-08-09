package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.User;
import java.util.List;

public record StaffUserView(
        String userId,
        String username,
        String displayName,
        List<String> roles
) {
    public static StaffUserView from(User user, List<String> roles) {
        return new StaffUserView(
                user.getUserId(),
                user.getUsername(),
                user.getDisplayName(),
                List.copyOf(roles)
        );
    }
}
