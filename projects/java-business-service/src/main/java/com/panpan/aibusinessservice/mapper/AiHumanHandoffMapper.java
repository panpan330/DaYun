package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.AiHumanHandoff;
import java.time.Instant;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface AiHumanHandoffMapper {
    int insert(AiHumanHandoff handoff);
    AiHumanHandoff findActiveByConversation(String conversationId);
    AiHumanHandoff findById(Long id);
    List<AiHumanHandoff> listByStatus(String status);
    int claim(@Param("id") Long id, @Param("agent") String agent, @Param("now") Instant now);
    int close(@Param("id") Long id, @Param("agent") String agent, @Param("note") String note, @Param("now") Instant now);
    int transfer(@Param("id") Long id, @Param("targetAgent") String targetAgent, @Param("note") String note, @Param("now") Instant now);
}
