package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.entity.AiConversation;
import com.panpan.aibusinessservice.entity.AiMessage;
import com.panpan.aibusinessservice.mapper.AiConversationMapper;
import com.panpan.aibusinessservice.mapper.AiMessageMapper;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

@SpringBootTest
class AiConversationMapperTest {

    @Autowired
    private AiConversationMapper conversationMapper;

    @Autowired
    private AiMessageMapper messageMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_messages");
        jdbcTemplate.update("DELETE FROM ai_conversations");
    }

    private AiConversation conversation(String conversationId, String userId, Instant updatedAt) {
        AiConversation c = new AiConversation();
        c.setTenantId("default");
        c.setConversationId(conversationId);
        c.setUserId(userId);
        c.setTitle("title-" + conversationId);
        c.setConversationStatus("active");
        c.setCreatedAt(updatedAt);
        c.setUpdatedAt(updatedAt);
        return c;
    }

    @Test
    void upsertIsIdempotentOnSameConversationId() {
        AiConversation first = conversation("conv-1", "U1001", Instant.now());
        conversationMapper.upsert(first);

        AiConversation second = conversation("conv-1", "U1001", Instant.now().plus(1, ChronoUnit.HOURS));
        second.setTitle("updated-title");
        conversationMapper.upsert(second);

        AiConversation loaded = conversationMapper.getByConversationId("default", "conv-1");
        assertNotNull(loaded);
        assertEquals("updated-title", loaded.getTitle());
        assertEquals(1, conversationMapper.listByUser("default", "U1001", 20).size());
    }

    @Test
    void listByUserOrdersByUpdatedAtDesc() {
        AiConversation old = conversation("conv-old", "U1001", Instant.now().minus(2, ChronoUnit.HOURS));
        AiConversation recent = conversation("conv-recent", "U1001", Instant.now());
        conversationMapper.upsert(old);
        conversationMapper.upsert(recent);

        List<AiConversation> list = conversationMapper.listByUser("default", "U1001", 20);
        assertEquals("conv-recent", list.get(0).getConversationId());
        assertEquals("conv-old", list.get(1).getConversationId());
    }

    @Test
    void listByUserRespectsLimit() {
        conversationMapper.upsert(conversation("conv-a", "U1001", Instant.now()));
        conversationMapper.upsert(conversation("conv-b", "U1001", Instant.now().plus(1, ChronoUnit.MINUTES)));
        conversationMapper.upsert(conversation("conv-c", "U1001", Instant.now().plus(2, ChronoUnit.MINUTES)));

        List<AiConversation> list = conversationMapper.listByUser("default", "U1001", 2);
        assertEquals(2, list.size());
        assertEquals("conv-c", list.get(0).getConversationId());
    }

    @Test
    void batchInsertIgnoresDuplicateMessageId() {
        Instant now = Instant.now();
        conversationMapper.upsert(conversation("conv-msg", "U1001", now));

        AiMessage msg = new AiMessage();
        msg.setTenantId("default");
        msg.setMessageId("msg-1");
        msg.setConversationId("conv-msg");
        msg.setSenderType("user");
        msg.setContent("hello");
        msg.setTraceId("trace-1");
        msg.setCreatedAt(now);

        assertEquals(1, messageMapper.batchInsert(List.of(msg)));
        assertEquals(0, messageMapper.batchInsert(List.of(msg)));

        assertEquals(1, messageMapper.countByConversation("default", "conv-msg"));
    }

    @Test
    void listByConversationOrdersByCreatedAtAsc() {
        Instant now = Instant.now();
        conversationMapper.upsert(conversation("conv-asc", "U1001", now));

        AiMessage older = message("msg-old", "conv-asc", "user", "first", now.minus(1, ChronoUnit.MINUTES));
        AiMessage newer = message("msg-new", "conv-asc", "assistant", "second", now);
        messageMapper.batchInsert(List.of(newer, older));

        List<AiMessage> list = messageMapper.listByConversation("default", "conv-asc");
        assertEquals(2, list.size());
        assertEquals("msg-old", list.get(0).getMessageId());
        assertEquals("msg-new", list.get(1).getMessageId());
    }

    @Test
    void deleteOlderThanRemovesOnlyOldConversations() {
        AiConversation fresh = conversation("conv-fresh", "U1001", Instant.now());
        AiConversation stale = conversation("conv-stale", "U1001", Instant.now().minus(31, ChronoUnit.DAYS));
        conversationMapper.upsert(fresh);
        conversationMapper.upsert(stale);

        int deleted = conversationMapper.deleteOlderThan(Instant.now().minus(30, ChronoUnit.DAYS));
        assertEquals(1, deleted);
        assertNull(conversationMapper.getByConversationId("default", "conv-stale"));
        assertNotNull(conversationMapper.getByConversationId("default", "conv-fresh"));
    }

    private AiMessage message(String messageId, String conversationId, String senderType, String content, Instant createdAt) {
        AiMessage msg = new AiMessage();
        msg.setTenantId("default");
        msg.setMessageId(messageId);
        msg.setConversationId(conversationId);
        msg.setSenderType(senderType);
        msg.setContent(content);
        msg.setTraceId("trace-" + messageId);
        msg.setCreatedAt(createdAt);
        return msg;
    }
}
