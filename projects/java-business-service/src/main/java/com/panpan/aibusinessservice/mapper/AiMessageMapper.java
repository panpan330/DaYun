package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.AiMessage;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface AiMessageMapper {

    int batchInsert(@Param("messages") List<AiMessage> messages);

    List<AiMessage> listByConversation(
            @Param("tenantId") String tenantId,
            @Param("conversationId") String conversationId
    );

    int countByConversation(
            @Param("tenantId") String tenantId,
            @Param("conversationId") String conversationId
    );
}
