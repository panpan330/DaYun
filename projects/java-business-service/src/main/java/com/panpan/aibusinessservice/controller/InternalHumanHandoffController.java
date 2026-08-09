package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.HumanHandoffView;
import com.panpan.aibusinessservice.dto.InternalCreateHandoffCommand;
import com.panpan.aibusinessservice.entity.AiHumanHandoff;
import com.panpan.aibusinessservice.mapper.AiHumanHandoffMapper;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.time.Instant;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ai-human-handoffs")
public class InternalHumanHandoffController {

    private final InternalRequestResolver requestResolver;
    private final AiHumanHandoffMapper handoffMapper;

    public InternalHumanHandoffController(
            InternalRequestResolver requestResolver,
            AiHumanHandoffMapper handoffMapper
    ) {
        this.requestResolver = requestResolver;
        this.handoffMapper = handoffMapper;
    }

    @PostMapping
    public ApiResponse<Void> create(
            @Valid @RequestBody InternalCreateHandoffCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        if (handoffMapper.findActiveByConversation(command.conversationId()) == null) {
            Instant now = Instant.now();
            AiHumanHandoff handoff = new AiHumanHandoff();
            handoff.setConversationId(command.conversationId());
            handoff.setUserId(command.userId());
            handoff.setTenantId(command.tenantId());
            handoff.setReason(command.reason());
            handoff.setRelatedOrderId(command.relatedOrderId());
            handoff.setEmotion(command.emotion());
            handoff.setStatus("pending");
            handoff.setCreatedAt(now);
            handoff.setUpdatedAt(now);
            handoffMapper.insert(handoff);
        }
        return ApiResponse.ok(null, context.traceId());
    }

    @GetMapping("/active-by-conversation")
    public ApiResponse<HumanHandoffView> activeByConversation(
            @RequestParam String conversationId,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        AiHumanHandoff handoff = handoffMapper.findActiveByConversation(conversationId);
        HumanHandoffView view = handoff == null ? null : toView(handoff);
        return ApiResponse.ok(view, context.traceId());
    }

    private HumanHandoffView toView(AiHumanHandoff h) {
        return new HumanHandoffView(
                h.getId(),
                h.getConversationId(),
                h.getUserId(),
                h.getReason(),
                h.getRelatedOrderId(),
                h.getEmotion(),
                h.getStatus(),
                h.getAssignedAgent(),
                h.getNote(),
                h.getCreatedAt(),
                h.getResolvedAt()
        );
    }
}
