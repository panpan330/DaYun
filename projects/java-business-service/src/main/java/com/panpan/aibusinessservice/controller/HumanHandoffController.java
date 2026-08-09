package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.HandoffCloseCommand;
import com.panpan.aibusinessservice.dto.HandoffTransferCommand;
import com.panpan.aibusinessservice.dto.HumanHandoffView;
import com.panpan.aibusinessservice.entity.AiHumanHandoff;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.AiHumanHandoffMapper;
import com.panpan.aibusinessservice.service.AuthService;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.time.Instant;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/human-handoffs")
public class HumanHandoffController {

    private final AuthService authService;
    private final AiHumanHandoffMapper handoffMapper;

    public HumanHandoffController(AuthService authService, AiHumanHandoffMapper handoffMapper) {
        this.authService = authService;
        this.handoffMapper = handoffMapper;
    }

    @GetMapping
    public ApiResponse<List<HumanHandoffView>> list(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @RequestParam(required = false) String status,
            HttpServletRequest servletRequest
    ) {
        authService.currentUser(authorization);
        List<AiHumanHandoff> rows = status == null || status.isBlank()
                ? handoffMapper.listByStatus("pending")
                : handoffMapper.listByStatus(status);
        List<HumanHandoffView> views = rows.stream().map(this::toView).toList();
        return ApiResponse.ok(views, TraceFilter.currentTraceId(servletRequest));
    }

    @PostMapping("/{id}/claim")
    public ApiResponse<HumanHandoffView> claim(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @PathVariable Long id,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        int updated = handoffMapper.claim(id, currentUser.userId(), Instant.now());
        if (updated == 0) {
            throw new BusinessException(BusinessErrorCode.HANDOFF_STATE_CONFLICT);
        }
        return ApiResponse.ok(toView(handoffMapper.findById(id)),
                TraceFilter.currentTraceId(servletRequest));
    }

    @PostMapping("/{id}/close")
    public ApiResponse<HumanHandoffView> close(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @PathVariable Long id,
            @Valid @RequestBody(required = false) HandoffCloseCommand command,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        String note = command == null ? null : command.note();
        int updated = handoffMapper.close(id, currentUser.userId(), note, Instant.now());
        if (updated == 0) {
            throw new BusinessException(BusinessErrorCode.HANDOFF_STATE_CONFLICT);
        }
        return ApiResponse.ok(toView(handoffMapper.findById(id)), TraceFilter.currentTraceId(servletRequest));
    }

    @PostMapping("/{id}/transfer")
    public ApiResponse<HumanHandoffView> transfer(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            @PathVariable Long id,
            @Valid @RequestBody HandoffTransferCommand command,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        AiHumanHandoff handoff = handoffMapper.findById(id);
        if (handoff == null || !"in_progress".equals(handoff.getStatus())) {
            throw new BusinessException(BusinessErrorCode.HANDOFF_STATE_CONFLICT);
        }
        boolean isCurrentAgent = currentUser.userId().equals(handoff.getAssignedAgent());
        boolean isSupervisor = currentUser.roles().contains("supervisor");
        if (!isCurrentAgent && !isSupervisor) {
            throw new BusinessException(BusinessErrorCode.HANDOFF_NOT_ASSIGNED);
        }
        if (command.targetAgent().equals(handoff.getAssignedAgent())) {
            throw new BusinessException(BusinessErrorCode.INVALID_REQUEST_PARAM);
        }
        String fromAgent = handoff.getAssignedAgent() == null ? "未分配" : handoff.getAssignedAgent();
        String appendedNote = (handoff.getNote() == null || handoff.getNote().isBlank())
                ? "由 " + fromAgent + " 转交 " + command.targetAgent()
                : handoff.getNote() + "；由 " + fromAgent + " 转交 " + command.targetAgent();
        int updated = handoffMapper.transfer(id, command.targetAgent(), appendedNote, Instant.now());
        if (updated == 0) {
            throw new BusinessException(BusinessErrorCode.HANDOFF_STATE_CONFLICT);
        }
        return ApiResponse.ok(toView(handoffMapper.findById(id)),
                TraceFilter.currentTraceId(servletRequest));
    }

    private HumanHandoffView toView(AiHumanHandoff handoff) {
        return new HumanHandoffView(
                handoff.getId(),
                handoff.getConversationId(),
                handoff.getUserId(),
                handoff.getReason(),
                handoff.getRelatedOrderId(),
                handoff.getEmotion(),
                handoff.getStatus(),
                handoff.getAssignedAgent(),
                handoff.getNote(),
                handoff.getCreatedAt(),
                handoff.getResolvedAt());
    }
}
