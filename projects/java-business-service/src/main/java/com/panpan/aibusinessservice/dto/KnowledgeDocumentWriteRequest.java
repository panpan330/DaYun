package com.panpan.aibusinessservice.dto;

public record KnowledgeDocumentWriteRequest(
        String documentId,
        String title,
        String docType,
        String businessDomain,
        String permissionGroup,
        String status,
        String sourceFileName,
        int chunkCount,
        String updatedBy
) {}
