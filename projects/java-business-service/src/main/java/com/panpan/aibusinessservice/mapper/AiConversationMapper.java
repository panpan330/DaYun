package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.AiConversation;
import java.time.Instant;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface AiConversationMapper {

    int upsert(AiConversation conversation);

    List<AiConversation> listByUser(
            @Param("tenantId") String tenantId,
            @Param("userId") String userId,
            @Param("limit") int limit
    );

    AiConversation getByConversationId(
            @Param("tenantId") String tenantId,
            @Param("conversationId") String conversationId
    );

    int deleteOlderThan(@Param("cutoff") Instant cutoff);
}
