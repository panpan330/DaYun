package com.panpan.aibusinessservice.dto;

import java.util.Map;

public record AiOpsStatsSummaryView(
        FeedbackStats feedback,
        HandoffStats handoffs,
        Map<String, Long> emotionDistribution,
        Map<String, Long> conversationVolume
) {
    public record FeedbackStats(long helpful, long unhelpful, Double helpfulRate) {}
    public record HandoffStats(long pending, long inProgress, long closed, long total) {}
}
