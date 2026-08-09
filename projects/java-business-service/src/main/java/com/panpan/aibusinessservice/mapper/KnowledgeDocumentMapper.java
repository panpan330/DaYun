package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.KnowledgeDocument;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface KnowledgeDocumentMapper {
    List<KnowledgeDocument> selectAllByTenantId(@Param("tenantId") String tenantId);

    List<KnowledgeDocument> selectVisibleByTenantIdAndPermissionGroups(
            @Param("tenantId") String tenantId,
            @Param("permissionGroups") List<String> permissionGroups
    );

    KnowledgeDocument selectByTenantIdAndDocumentId(
            @Param("tenantId") String tenantId,
            @Param("documentId") String documentId
    );

    int upsert(KnowledgeDocument entity);

    int deleteByTenantIdAndDocumentId(
            @Param("tenantId") String tenantId,
            @Param("documentId") String documentId
    );
}
