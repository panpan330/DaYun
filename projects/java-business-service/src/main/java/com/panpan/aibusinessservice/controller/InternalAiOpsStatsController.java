package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.AiOpsStatsSummaryView;
import com.panpan.aibusinessservice.dto.AiOpsStatsSummaryView.FeedbackStats;
import com.panpan.aibusinessservice.dto.AiOpsStatsSummaryView.HandoffStats;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.DayCount;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.EmotionCount;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.RatingCount;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.StatusCount;
import jakarta.servlet.http.HttpServletRequest;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ai-ops-stats")
public class InternalAiOpsStatsController {

    private final InternalRequestResolver requestResolver;
    private final AiOpsStatsMapper mapper;

    public InternalAiOpsStatsController(InternalRequestResolver requestResolver, AiOpsStatsMapper mapper) {
        this.requestResolver = requestResolver;
        this.mapper = mapper;
    }

    @GetMapping("/summary")
    public ApiResponse<AiOpsStatsSummaryView> summary(
            @RequestParam(defaultValue = "7") int days,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        if (days != 7 && days != 30) {
            throw new BusinessException(BusinessErrorCode.INVALID_REQUEST_PARAM);
        }
        LocalDateTime cutoff = LocalDateTime.now().minusDays(days);

        long helpful = ratingCount(mapper.countFeedbackByRating(cutoff), "helpful");
        long unhelpful = ratingCount(mapper.countFeedbackByRating(cutoff), "unhelpful");
        long total = helpful + unhelpful;
        Double helpfulRate = total == 0 ? null : (double) helpful / total;

        long pending = statusCount(mapper.countHandoffsByStatus(cutoff), "pending");
        long inProgress = statusCount(mapper.countHandoffsByStatus(cutoff), "in_progress");
        long closed = statusCount(mapper.countHandoffsByStatus(cutoff), "closed");

        Map<String, Long> emotion = new LinkedHashMap<>();
        for (EmotionCount row : mapper.countHandoffsByEmotion(cutoff)) {
            emotion.put(row.emotion(), row.count());
        }
        Map<String, Long> volume = new LinkedHashMap<>();
        for (DayCount row : mapper.countUserMessagesByDay(cutoff)) {
            volume.put(row.statDate().toString(), row.count());
        }

        AiOpsStatsSummaryView view = new AiOpsStatsSummaryView(
                new FeedbackStats(helpful, unhelpful, helpfulRate),
                new HandoffStats(pending, inProgress, closed, pending + inProgress + closed),
                emotion,
                volume
        );
        return ApiResponse.ok(view, context.traceId());
    }

    private long ratingCount(List<RatingCount> rows, String key) {
        return rows.stream().filter(r -> key.equals(r.rating())).mapToLong(RatingCount::count).findFirst().orElse(0L);
    }

    private long statusCount(List<StatusCount> rows, String key) {
        return rows.stream().filter(r -> key.equals(r.status())).mapToLong(StatusCount::count).findFirst().orElse(0L);
    }
}
