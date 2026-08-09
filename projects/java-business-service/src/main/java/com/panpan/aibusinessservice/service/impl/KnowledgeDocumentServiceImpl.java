package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentWriteRequest;
import com.panpan.aibusinessservice.entity.KnowledgeDocument;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.KnowledgeDocumentMapper;
import com.panpan.aibusinessservice.service.KnowledgeDocumentService;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.springframework.stereotype.Service;

@Service
public class KnowledgeDocumentServiceImpl implements KnowledgeDocumentService {
    private final KnowledgeDocumentMapper knowledgeDocumentMapper;

    public KnowledgeDocumentServiceImpl(KnowledgeDocumentMapper knowledgeDocumentMapper) {
        this.knowledgeDocumentMapper = knowledgeDocumentMapper;
    }

    @Override
    public List<KnowledgeDocumentView> listVisibleDocuments(CurrentUserView currentUser) {
        if (currentUser.roles().contains("admin")) {
            return knowledgeDocumentMapper.selectAllByTenantId(currentUser.tenantId())
                    .stream()
                    .map(KnowledgeDocumentView::from)
                    .toList();
        }

        List<String> permissionGroups = permissionGroupsFor(currentUser.roles());
        return knowledgeDocumentMapper
                .selectVisibleByTenantIdAndPermissionGroups(currentUser.tenantId(), permissionGroups)
                .stream()
                .map(KnowledgeDocumentView::from)
                .toList();
    }

    private List<String> permissionGroupsFor(List<String> roles) {
        Set<String> permissionGroups = new LinkedHashSet<>();
        permissionGroups.add("public");

        if (roles.contains("customer")) {
            permissionGroups.add("customer");
        }
        if (roles.contains("agent") || roles.contains("supervisor")) {
            permissionGroups.add("customer_service");
        }

        return List.copyOf(permissionGroups);
    }

    @Override
    public List<KnowledgeDocumentView> listAllDocuments(String tenantId) {
        return knowledgeDocumentMapper.selectAllByTenantId(tenantId)
                .stream()
                .map(KnowledgeDocumentView::from)
                .toList();
    }

    @Override
    public KnowledgeDocumentView upsertDocument(
            InternalRequestContext context,
            KnowledgeDocumentWriteRequest request,
            String idempotencyKey) {
        if (request.documentId() == null || request.documentId().isBlank()) {
            throw new BusinessException(BusinessErrorCode.DOCUMENT_ID_REQUIRED);
        }
        if (request.title() == null || request.title().isBlank()) {
            throw new BusinessException(BusinessErrorCode.DOCUMENT_TITLE_REQUIRED);
        }

        KnowledgeDocument entity = new KnowledgeDocument();
        entity.setTenantId(context.tenantId());
        entity.setDocumentId(request.documentId());
        entity.setTitle(request.title());
        entity.setDocType(request.docType() == null ? "policy" : request.docType());
        entity.setBusinessDomain(
                request.businessDomain() == null ? "general" : request.businessDomain());
        entity.setPermissionGroup(
                request.permissionGroup() == null ? "public" : request.permissionGroup());
        entity.setStatus(request.status() == null ? "enabled" : request.status());
        entity.setSourceFileName(request.sourceFileName());
        entity.setChunkCount(request.chunkCount());
        entity.setUpdatedBy(request.updatedBy() == null ? context.userId() : request.updatedBy());

        knowledgeDocumentMapper.upsert(entity);
        return KnowledgeDocumentView.from(entity);
    }

    @Override
    public boolean deleteDocument(InternalRequestContext context, String documentId) {
        if (documentId == null || documentId.isBlank()) {
            throw new BusinessException(BusinessErrorCode.DOCUMENT_ID_REQUIRED);
        }
        return knowledgeDocumentMapper.deleteByTenantIdAndDocumentId(context.tenantId(), documentId) > 0;
    }
}
