package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.AiConversationView;
import com.panpan.aibusinessservice.dto.AiMessageView;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesCommand;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesReceipt;
import com.panpan.aibusinessservice.dto.CleanupAiConversationsReceipt;
import com.panpan.aibusinessservice.dto.UpsertAiConversationCommand;
import com.panpan.aibusinessservice.service.AiConversationService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ai-conversations")
public class InternalAiConversationController {
    private final InternalRequestResolver requestResolver;
    private final AiConversationService conversationService;

    public InternalAiConversationController(
            InternalRequestResolver requestResolver,
            AiConversationService conversationService
    ) {
        this.requestResolver = requestResolver;
        this.conversationService = conversationService;
    }

    @PostMapping
    public ApiResponse<Void> upsert(
            @Valid @RequestBody UpsertAiConversationCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        conversationService.upsert(command, context);
        return ApiResponse.ok(null, context.traceId());
    }

    @PostMapping("/messages")
    public ApiResponse<BatchWriteAiMessagesReceipt> batchWriteMessages(
            @Valid @RequestBody BatchWriteAiMessagesCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                conversationService.batchWriteMessages(command, context),
                context.traceId()
        );
    }

    @GetMapping
    public ApiResponse<List<AiConversationView>> listByUser(
            @RequestParam(defaultValue = "20") int limit,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                conversationService.listByUser(context.tenantId(), context.userId(), limit),
                context.traceId()
        );
    }

    @GetMapping("/{conversationId}/messages")
    public ApiResponse<List<AiMessageView>> getMessages(
            @PathVariable String conversationId,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                conversationService.getMessages(context.tenantId(), conversationId),
                context.traceId()
        );
    }

    @PostMapping("/cleanup")
    public ApiResponse<CleanupAiConversationsReceipt> cleanup(
            @RequestParam(defaultValue = "30") int olderThanDays,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                conversationService.cleanupOlderThanDays(olderThanDays),
                context.traceId()
        );
    }
}
