package com.panpan.aibusinessservice.mapper;

import java.time.LocalDateTime;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface AiOpsStatsMapper {
    record RatingCount(String rating, long count) {}
    record StatusCount(String status, long count) {}
    record EmotionCount(String emotion, long count) {}
    record DayCount(java.time.LocalDate statDate, long count) {}

    List<RatingCount> countFeedbackByRating(@Param("cutoff") LocalDateTime cutoff);
    List<StatusCount> countHandoffsByStatus(@Param("cutoff") LocalDateTime cutoff);
    List<EmotionCount> countHandoffsByEmotion(@Param("cutoff") LocalDateTime cutoff);
    List<DayCount> countUserMessagesByDay(@Param("cutoff") LocalDateTime cutoff);
}
