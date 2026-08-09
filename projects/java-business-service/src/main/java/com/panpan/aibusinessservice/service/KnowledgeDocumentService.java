package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentWriteRequest;
import java.util.List;

public interface KnowledgeDocumentService {
    List<KnowledgeDocumentView> listVisibleDocuments(CurrentUserView currentUser);

    List<KnowledgeDocumentView> listAllDocuments(String tenantId);

    KnowledgeDocumentView upsertDocument(
            InternalRequestContext context,
            KnowledgeDocumentWriteRequest request,
            String idempotencyKey);

    boolean deleteDocument(InternalRequestContext context, String documentId);
}
