package com.panpan.aibusinessservice;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.DayCount;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.EmotionCount;
import com.panpan.aibusinessservice.mapper.AiOpsStatsMapper.RatingCount;

@SpringBootTest
class AiOpsStatsMapperTest {

    @Autowired
    private AiOpsStatsMapper mapper;

    @Autowired
    private JdbcTemplate jdbc;

    @BeforeEach
    void clean() {
        jdbc.update("DELETE FROM ai_response_feedback");
        jdbc.update("DELETE FROM ai_human_handoffs");
        jdbc.update("DELETE FROM ai_messages");
    }

    @Test
    void countFeedbackByRating_countsRecentRows() {
        insertFeedback("helpful", "2026-08-07 10:00:00");
        insertFeedback("helpful", "2026-08-06 10:00:00");
        insertFeedback("unhelpful", "2026-08-07 11:00:00");
        insertFeedback("unhelpful", "2026-07-01 10:00:00");

        List<RatingCount> rows = mapper.countFeedbackByRating(LocalDateTime.now().minusDays(7));
        Map<String, Long> byRating = rows.stream().collect(
                Collectors.toMap(RatingCount::rating, RatingCount::count));
        assertEquals(2L, byRating.get("helpful"));
        assertEquals(1L, byRating.get("unhelpful"));
    }

    @Test
    void countUserMessagesByDay_groupsByDay() {
        jdbc.update(
                "INSERT INTO ai_messages (tenant_id, message_id, conversation_id, sender_type, content, trace_id, created_at) "
                        + "VALUES (?, ?, ?, 'user', 'hi', 't', ?)",
                "default", "m1", "c1", "2026-08-07 09:00:00");
        jdbc.update(
                "INSERT INTO ai_messages (tenant_id, message_id, conversation_id, sender_type, content, trace_id, created_at) "
                        + "VALUES (?, ?, ?, 'user', 'hi', 't', ?)",
                "default", "m2", "c1", "2026-08-07 18:00:00");
        jdbc.update(
                "INSERT INTO ai_messages (tenant_id, message_id, conversation_id, sender_type, content, trace_id, created_at) "
                        + "VALUES (?, ?, ?, 'assistant', 'ok', 't', ?)",
                "default", "m3", "c1", "2026-08-07 18:01:00");

        List<DayCount> rows = mapper.countUserMessagesByDay(LocalDateTime.now().minusDays(7));
        assertEquals(1, rows.size());
        assertEquals("2026-08-07", rows.get(0).statDate().toString());
        assertEquals(2L, rows.get(0).count());
    }

    @Test
    void countHandoffsByEmotion_countsNullAsUnknownAndNormalizesEnumPrefix() {
        insertHandoff("c1", "pending", "angry", "2026-08-07 10:00:00");
        insertHandoff("c2", "closed", null, "2026-08-07 11:00:00");
        insertHandoff("c3", "closed", "CustomerEmotion.ANGRY", "2026-08-07 12:00:00");

        List<EmotionCount> rows = mapper.countHandoffsByEmotion(LocalDateTime.now().minusDays(7));
        Map<String, Long> byEmotion = rows.stream().collect(
                Collectors.toMap(EmotionCount::emotion, EmotionCount::count));
        // 'angry' 与 'CustomerEmotion.ANGRY' 归一化为同一键 'angry'
        assertEquals(2L, byEmotion.get("angry"));
        assertEquals(1L, byEmotion.get("unknown"));
    }

    private void insertFeedback(String rating, String createdAt) {
        jdbc.update(
                "INSERT INTO ai_response_feedback "
                        + "(tenant_id, user_id, conversation_id, trace_id, rating, agent_route, created_at, updated_at) "
                        + "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                "default", "U1", "conv-" + System.nanoTime(), "t-" + System.nanoTime(),
                rating, "order_query", createdAt, createdAt);
    }

    private void insertHandoff(String conversationId, String status, String emotion, String createdAt) {
        jdbc.update(
                "INSERT INTO ai_human_handoffs "
                        + "(conversation_id, user_id, tenant_id, reason, status, emotion, created_at, updated_at) "
                        + "VALUES (?, ?, ?, 'r', ?, ?, ?, ?)",
                conversationId, "U1", "default", status, emotion, createdAt, createdAt);
    }
}
