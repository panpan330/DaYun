package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.StaffUserView;
import com.panpan.aibusinessservice.service.AuthService;
import com.panpan.aibusinessservice.service.UserDirectoryService;
import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class UserController {
    private final AuthService authService;
    private final UserDirectoryService userDirectoryService;

    public UserController(AuthService authService, UserDirectoryService userDirectoryService) {
        this.authService = authService;
        this.userDirectoryService = userDirectoryService;
    }

    @GetMapping("/staff")
    public ApiResponse<List<StaffUserView>> listStaff(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        return ApiResponse.ok(
                userDirectoryService.listAssignableStaff(currentUser),
                TraceFilter.currentTraceId(servletRequest)
        );
    }
}
