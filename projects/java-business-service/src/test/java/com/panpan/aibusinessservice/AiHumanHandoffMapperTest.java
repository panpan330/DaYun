package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.entity.AiHumanHandoff;
import com.panpan.aibusinessservice.mapper.AiHumanHandoffMapper;
import java.time.Instant;
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
class AiHumanHandoffMapperTest {

    @Autowired
    private AiHumanHandoffMapper mapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_human_handoffs");
    }

    private AiHumanHandoff sample(String conversationId) {
        Instant now = Instant.now();
        AiHumanHandoff h = new AiHumanHandoff();
        h.setConversationId(conversationId);
        h.setUserId("U1001");
        h.setTenantId("default");
        h.setReason("检测到强烈情绪（angry），建议由人工客服继续跟进。");
        h.setRelatedOrderId("202501010001");
        h.setEmotion("angry");
        h.setStatus("pending");
        h.setCreatedAt(now);
        h.setUpdatedAt(now);
        return h;
    }

    @Test
    void insertAndFindActive() {
        mapper.insert(sample("conv-1"));
        AiHumanHandoff active = mapper.findActiveByConversation("conv-1");
        assertNotNull(active);
        assertEquals("pending", active.getStatus());
    }

    @Test
    void claimAndCloseTransitionsStatus() {
        mapper.insert(sample("conv-2"));
        AiHumanHandoff active = mapper.findActiveByConversation("conv-2");
        mapper.claim(active.getId(), "agent-A", Instant.now());
        assertEquals("in_progress", mapper.findActiveByConversation("conv-2").getStatus());
        mapper.close(active.getId(), "agent-A", "已电话联系客户", Instant.now());
        assertNull(mapper.findActiveByConversation("conv-2"));
        assertEquals(1, mapper.listByStatus("closed").size());
    }

    @Test
    void listByStatusFilters() {
        mapper.insert(sample("conv-3"));
        mapper.insert(sample("conv-4"));
        List<AiHumanHandoff> pending = mapper.listByStatus("pending");
        assertEquals(2, pending.size());
        assertEquals(0, mapper.listByStatus("in_progress").size());
    }
}
