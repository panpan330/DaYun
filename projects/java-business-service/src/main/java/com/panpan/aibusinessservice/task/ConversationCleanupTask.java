package com.panpan.aibusinessservice.task;

import com.panpan.aibusinessservice.service.AiConversationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class ConversationCleanupTask {
    private static final Logger log = LoggerFactory.getLogger(ConversationCleanupTask.class);
    private static final int RETENTION_DAYS = 30;

    private final AiConversationService conversationService;

    public ConversationCleanupTask(AiConversationService conversationService) {
        this.conversationService = conversationService;
    }

    @Scheduled(cron = "0 0 3 * * *")
    public void cleanupExpiredConversations() {
        try {
            int deleted = conversationService.cleanupOlderThanDays(RETENTION_DAYS).deleted();
            if (deleted > 0) {
                log.info("ai conversation cleanup: deleted {} conversations older than {} days", deleted, RETENTION_DAYS);
            }
        } catch (Exception exc) {
            log.error("ai conversation cleanup failed", exc);
        }
    }
}
