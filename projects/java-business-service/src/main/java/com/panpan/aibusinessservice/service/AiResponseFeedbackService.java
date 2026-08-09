package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackOverviewView;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackContextView;
import com.panpan.aibusinessservice.dto.AiResponseFeedbackReceipt;
import com.panpan.aibusinessservice.dto.CreateAiResponseFeedbackCommand;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.PromoteAiFeedbackBadCaseCommand;
import com.panpan.aibusinessservice.dto.ReviewAiFeedbackCommand;

public interface AiResponseFeedbackService {
    AiResponseFeedbackReceipt upsert(CreateAiResponseFeedbackCommand command, InternalRequestContext context);

    AiResponseFeedbackOverviewView getOverview(CurrentUserView currentUser);

    AiResponseFeedbackContextView getInternalContext(long feedbackId, InternalRequestContext context);

    AiResponseFeedbackContextView markBadCasePromoted(
            long feedbackId,
            PromoteAiFeedbackBadCaseCommand command,
            InternalRequestContext context
    );

    AiResponseFeedbackContextView review(
            long feedbackId,
            ReviewAiFeedbackCommand command,
            InternalRequestContext context
    );
}
