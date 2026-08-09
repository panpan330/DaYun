package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackOverviewView;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.service.AiResponseFeedbackService;
import com.panpan.aibusinessservice.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ai-response-feedback")
public class AiResponseFeedbackController {
    private final AuthService authService;
    private final AiResponseFeedbackService feedbackService;

    public AiResponseFeedbackController(AuthService authService, AiResponseFeedbackService feedbackService) {
        this.authService = authService;
        this.feedbackService = feedbackService;
    }

    @GetMapping("/overview")
    public ApiResponse<AiResponseFeedbackOverviewView> getOverview(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest request
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        return ApiResponse.ok(
                feedbackService.getOverview(currentUser),
                TraceFilter.currentTraceId(request)
        );
    }
}
