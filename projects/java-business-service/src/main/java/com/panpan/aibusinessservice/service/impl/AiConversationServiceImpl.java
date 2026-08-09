package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.AiConversationView;
import com.panpan.aibusinessservice.dto.AiMessageView;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesCommand;
import com.panpan.aibusinessservice.dto.BatchWriteAiMessagesReceipt;
import com.panpan.aibusinessservice.dto.CleanupAiConversationsReceipt;
import com.panpan.aibusinessservice.dto.UpsertAiConversationCommand;
import com.panpan.aibusinessservice.entity.AiConversation;
import com.panpan.aibusinessservice.entity.AiMessage;
import com.panpan.aibusinessservice.mapper.AiConversationMapper;
import com.panpan.aibusinessservice.mapper.AiMessageMapper;
import com.panpan.aibusinessservice.service.AiConversationService;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class AiConversationServiceImpl implements AiConversationService {

    private static final int DEFAULT_LIST_LIMIT = 20;

    private final AiConversationMapper conversationMapper;
    private final AiMessageMapper messageMapper;

    public AiConversationServiceImpl(AiConversationMapper conversationMapper, AiMessageMapper messageMapper) {
        this.conversationMapper = conversationMapper;
        this.messageMapper = messageMapper;
    }

    @Override
    @Transactional
    public void upsert(UpsertAiConversationCommand command, InternalRequestContext context) {
        AiConversation conversation = new AiConversation();
        conversation.setTenantId(context.tenantId());
        conversation.setConversationId(command.conversationId());
        conversation.setUserId(command.userId());
        conversation.setTitle(command.title());
        conversation.setConversationStatus(command.conversationStatus());
        Instant now = Instant.now();
        conversation.setCreatedAt(now);
        conversation.setUpdatedAt(now);
        conversationMapper.upsert(conversation);
    }

    @Override
    @Transactional
    public BatchWriteAiMessagesReceipt batchWriteMessages(BatchWriteAiMessagesCommand command, InternalRequestContext context) {
        Instant now = Instant.now();
        List<AiMessage> messages = command.messages().stream()
                .map(payload -> {
                    AiMessage message = new AiMessage();
                    message.setTenantId(context.tenantId());
                    message.setMessageId(payload.messageId());
                    message.setConversationId(command.conversationId());
                    message.setSenderType(payload.senderType());
                    message.setContent(payload.content());
                    message.setTraceId(payload.traceId());
                    message.setCreatedAt(now);
                    return message;
                })
                .toList();
        int inserted = messageMapper.batchInsert(messages);
        return new BatchWriteAiMessagesReceipt(inserted);
    }

    @Override
    public List<AiConversationView> listByUser(String tenantId, String userId, int limit) {
        int effectiveLimit = limit > 0 ? Math.min(limit, 100) : DEFAULT_LIST_LIMIT;
        return conversationMapper.listByUser(tenantId, userId, effectiveLimit).stream()
                .map(c -> new AiConversationView(
                        c.getConversationId(),
                        c.getUserId(),
                        c.getTitle(),
                        c.getConversationStatus(),
                        c.getUpdatedAt()))
                .toList();
    }

    @Override
    public List<AiMessageView> getMessages(String tenantId, String conversationId) {
        return messageMapper.listByConversation(tenantId, conversationId).stream()
                .map(m -> new AiMessageView(
                        m.getMessageId(),
                        m.getConversationId(),
                        m.getSenderType(),
                        m.getContent(),
                        m.getTraceId(),
                        m.getCreatedAt()))
                .toList();
    }

    @Override
    @Transactional
    public CleanupAiConversationsReceipt cleanupOlderThanDays(int days) {
        Instant cutoff = Instant.now().minus(days, ChronoUnit.DAYS);
        return new CleanupAiConversationsReceipt(conversationMapper.deleteOlderThan(cutoff));
    }
}
