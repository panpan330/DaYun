package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.AiConversationView;
import com.panpan.aibusinessservice.dto.AiMessageView;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesCommand;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesReceipt;
import com.panpan.aibusinessservice.dto.CleanupAiConversationsReceipt;
import com.panpan.aibusinessservice.dto.UpsertAiConversationCommand;
import java.util.List;

public interface AiConversationService {

    void upsert(UpsertAiConversationCommand command, InternalRequestContext context);

    BatchWriteAiMessagesReceipt batchWriteMessages(BatchWriteAiMessagesCommand command, InternalRequestContext context);

    List<AiConversationView> listByUser(String tenantId, String userId, int limit);

    List<AiMessageView> getMessages(String tenantId, String conversationId);

    CleanupAiConversationsReceipt cleanupOlderThanDays(int days);
}
