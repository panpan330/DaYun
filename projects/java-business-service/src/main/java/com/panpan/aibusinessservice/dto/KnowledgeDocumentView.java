package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.entity.KnowledgeDocument;
import java.time.LocalDateTime;

public record KnowledgeDocumentView(
        String documentId,
        String title,
        String docType,
        String businessDomain,
        String permissionGroup,
        String status,
        String sourceFileName,
        int chunkCount,
        String updatedBy,
        LocalDateTime updatedAt
) {
    public static KnowledgeDocumentView from(KnowledgeDocument document) {
        return new KnowledgeDocumentView(
                document.getDocumentId(),
                document.getTitle(),
                document.getDocType(),
                document.getBusinessDomain(),
                document.getPermissionGroup(),
                document.getStatus(),
                document.getSourceFileName(),
                document.getChunkCount(),
                document.getUpdatedBy(),
                document.getUpdatedAt()
        );
    }
}
