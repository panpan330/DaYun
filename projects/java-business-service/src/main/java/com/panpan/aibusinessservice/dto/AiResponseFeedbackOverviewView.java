package com.panpan.aibusinessservice.dto;

import java.util.List;

public record AiResponseFeedbackOverviewView(
        long totalCount,
        long helpfulCount,
        long unhelpfulCount,
        double unhelpfulRate,
        List<AiFeedbackReasonCountView> reasonCounts,
        List<AiFeedbackRegressionCandidateView> regressionCandidates
) {}
