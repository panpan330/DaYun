package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.User;
import java.util.List;

public record CurrentUserView(
        String tenantId,
        String userId,
        String username,
        String displayName,
        List<String> roles,
        String defaultHomePath
) {
    public static CurrentUserView from(User user, List<String> roles) {
        return new CurrentUserView(
                user.getTenantId(),
                user.getUserId(),
                user.getUsername(),
                user.getDisplayName(),
                List.copyOf(roles),
                defaultHomePath(roles)
        );
    }

    private static String defaultHomePath(List<String> roles) {
        if (roles.contains("admin")) {
            return "/dashboard";
        }
        if (roles.contains("supervisor")) {
            return "/knowledge";
        }
        if (roles.contains("agent")) {
            return "/workbench";
        }
        return "/ai-chat";
    }
}
