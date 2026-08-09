package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.AiFeedbackReasonCountView;
import com.panpan.aibusinessservice.dto.AiFeedbackRegressionCandidateView;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackOverviewView;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackContextView;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackReceipt;
import com.panpan.aibusinessservice.dto.CreateAiResponseFeedbackCommand;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.PromoteAiFeedbackBadCaseCommand;
import com.panpan.aibusinessservice.dto.ReviewAiFeedbackCommand;
import com.panpan.aibusinessservice.entity.AiResponseFeedback;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.AiResponseFeedbackMapper;
import com.panpan.aibusinessservice.service.AiResponseFeedbackService;
import java.time.Instant;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AiResponseFeedbackServiceImpl implements AiResponseFeedbackService {
    private static final int RECENT_CANDIDATE_LIMIT = 20;
    private final AiResponseFeedbackMapper feedbackMapper;

    public AiResponseFeedbackServiceImpl(AiResponseFeedbackMapper feedbackMapper) {
        this.feedbackMapper = feedbackMapper;
    }

    @Override
    @Transactional
    public AiResponseFeedbackReceipt upsert(CreateAiResponseFeedbackCommand command, InternalRequestContext context) {
        Instant now = Instant.now();
        AiResponseFeedback feedback = feedbackMapper.selectByIdentity(
                context.tenantId(), context.userId(), command.conversationId(), command.traceId()
        );
        if (feedback == null) {
            feedback = new AiResponseFeedback();
            feedback.setTenantId(context.tenantId());
            feedback.setUserId(context.userId());
            feedback.setConversationId(command.conversationId());
            feedback.setTraceId(command.traceId());
            feedback.setCreatedAt(now);
        }
        feedback.setRating(command.rating());
        feedback.setReason("helpful".equals(command.rating()) ? null : blankToNull(command.reason()));
        feedback.setAgentRoute(command.agentRoute());
        feedback.setCitationCount(command.citationCount());
        feedback.setHumanHandoffSuggested(command.humanHandoffSuggested());
        feedback.setUserMessageExcerpt(blankToNull(command.userMessageExcerpt()));
        feedback.setAssistantAnswerExcerpt(blankToNull(command.assistantAnswerExcerpt()));
        feedback.setCitationSummaryJson(blankToNull(command.citationSummaryJson()));
        if (feedback.getReviewStatus() == null || feedback.getReviewStatus().isBlank()) {
            feedback.setReviewStatus("candidate");
        }
        feedback.setUpdatedAt(now);

        if (feedback.getId() == null) {
            feedbackMapper.insert(feedback);
        } else {
            feedbackMapper.update(feedback);
        }
        return AiResponseFeedbackReceipt.from(feedback);
    }

    @Override
    public AiResponseFeedbackOverviewView getOverview(CurrentUserView currentUser) {
        if (!currentUser.roles().contains("supervisor") && !currentUser.roles().contains("admin")) {
            throw new BusinessException(BusinessErrorCode.AI_FEEDBACK_ACCESS_DENIED);
        }
        String tenantId = currentUser.tenantId();
        long helpful = feedbackMapper.countByTenantAndRating(tenantId, "helpful");
        long unhelpful = feedbackMapper.countByTenantAndRating(tenantId, "unhelpful");
        long total = helpful + unhelpful;
        List<AiFeedbackReasonCountView> reasonCounts = feedbackMapper.selectReasonCountsByTenantId(tenantId)
                .stream()
                .map(item -> new AiFeedbackReasonCountView(
                        item.getReason() == null ? "unspecified" : item.getReason(),
                        item.getCitationCount()
                ))
                .toList();
        List<AiFeedbackRegressionCandidateView> candidates = feedbackMapper
                .selectRecentUnhelpfulByTenantId(tenantId, RECENT_CANDIDATE_LIMIT)
                .stream()
                .map(AiFeedbackRegressionCandidateView::from)
                .toList();
        return new AiResponseFeedbackOverviewView(
                total, helpful, unhelpful, total == 0 ? 0 : (double) unhelpful / total,
                reasonCounts, candidates
        );
    }

    @Override
    public AiResponseFeedbackContextView getInternalContext(long feedbackId, InternalRequestContext context) {
        return AiResponseFeedbackContextView.from(requireFeedback(context.tenantId(), feedbackId));
    }

    @Override
    @Transactional
    public AiResponseFeedbackContextView markBadCasePromoted(
            long feedbackId,
            PromoteAiFeedbackBadCaseCommand command,
            InternalRequestContext context
    ) {
        AiResponseFeedback feedback = requireFeedback(context.tenantId(), feedbackId);
        feedbackMapper.markBadCasePromoted(
                context.tenantId(), feedbackId, command.badCaseId().trim(), context.userId(), Instant.now()
        );
        feedback.setReviewStatus("regression_added");
        feedback.setBadCaseId(command.badCaseId().trim());
        return AiResponseFeedbackContextView.from(feedback);
    }

    private AiResponseFeedback requireFeedback(String tenantId, long feedbackId) {
        AiResponseFeedback feedback = feedbackMapper.selectByTenantIdAndId(tenantId, feedbackId);
        if (feedback == null) {
            throw new BusinessException(BusinessErrorCode.AI_FEEDBACK_NOT_FOUND);
        }
        return feedback;
    }

    @Override
    @Transactional
    public AiResponseFeedbackContextView review(
            long feedbackId,
            ReviewAiFeedbackCommand command,
            InternalRequestContext context
    ) {
        AiResponseFeedback feedback = requireFeedback(context.tenantId(), feedbackId);
        Instant now = Instant.now();
        String note = blankToNull(command.reviewNote());
        feedbackMapper.updateReview(
                context.tenantId(), feedbackId, command.reviewStatus(), note, context.userId(), now
        );
        feedback.setReviewStatus(command.reviewStatus());
        feedback.setReviewNote(note);
        feedback.setReviewedByUserId(context.userId());
        feedback.setReviewedAt(now);
        return AiResponseFeedbackContextView.from(feedback);
    }

    private String blankToNull(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }
}
