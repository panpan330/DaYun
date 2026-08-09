package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackReceipt;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackContextView;
import com.panpan.aibusinessservice.dto.CreateAiResponseFeedbackCommand;
import com.panpan.aibusinessservice.dto.PromoteAiFeedbackBadCaseCommand;
import com.panpan.aibusinessservice.dto.ReviewAiFeedbackCommand;
import com.panpan.aibusinessservice.service.AiResponseFeedbackService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ai-response-feedback")
public class InternalAiResponseFeedbackController {
    private final InternalRequestResolver requestResolver;
    private final AiResponseFeedbackService feedbackService;

    public InternalAiResponseFeedbackController(
            InternalRequestResolver requestResolver,
            AiResponseFeedbackService feedbackService
    ) {
        this.requestResolver = requestResolver;
        this.feedbackService = feedbackService;
    }

    @PostMapping
    public ApiResponse<AiResponseFeedbackReceipt> upsert(
            @Valid @RequestBody CreateAiResponseFeedbackCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(feedbackService.upsert(command, context), context.traceId());
    }

    @GetMapping("/{feedbackId}")
    public ApiResponse<AiResponseFeedbackContextView> getContext(
            @PathVariable long feedbackId,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(feedbackService.getInternalContext(feedbackId, context), context.traceId());
    }

    @PostMapping("/{feedbackId}/promote")
    public ApiResponse<AiResponseFeedbackContextView> promote(
            @PathVariable long feedbackId,
            @Valid @RequestBody PromoteAiFeedbackBadCaseCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                feedbackService.markBadCasePromoted(feedbackId, command, context),
                context.traceId()
        );
    }

    @PostMapping("/{feedbackId}/review")
    public ApiResponse<AiResponseFeedbackContextView> review(
            @PathVariable long feedbackId,
            @Valid @RequestBody ReviewAiFeedbackCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(feedbackService.review(feedbackId, command, context), context.traceId());
    }
}
