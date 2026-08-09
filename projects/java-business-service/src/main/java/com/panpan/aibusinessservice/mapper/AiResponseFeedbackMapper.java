package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.AiResponseFeedback;
import java.time.Instant;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface AiResponseFeedbackMapper {
    AiResponseFeedback selectByIdentity(
            @Param("tenantId") String tenantId,
            @Param("userId") String userId,
            @Param("conversationId") String conversationId,
            @Param("traceId") String traceId
    );

    AiResponseFeedback selectByTenantIdAndId(
            @Param("tenantId") String tenantId,
            @Param("id") long id
    );

    int insert(AiResponseFeedback feedback);

    int update(AiResponseFeedback feedback);

    int markBadCasePromoted(
            @Param("tenantId") String tenantId,
            @Param("id") long id,
            @Param("badCaseId") String badCaseId,
            @Param("reviewedByUserId") String reviewedByUserId,
            @Param("reviewedAt") Instant reviewedAt
    );

    int updateReview(
            @Param("tenantId") String tenantId,
            @Param("id") long id,
            @Param("reviewStatus") String reviewStatus,
            @Param("reviewNote") String reviewNote,
            @Param("reviewedByUserId") String reviewedByUserId,
            @Param("reviewedAt") Instant reviewedAt
    );

    long countByTenantAndRating(@Param("tenantId") String tenantId, @Param("rating") String rating);

    List<AiResponseFeedback> selectReasonCountsByTenantId(@Param("tenantId") String tenantId);

    List<AiResponseFeedback> selectRecentUnhelpfulByTenantId(
            @Param("tenantId") String tenantId,
            @Param("limit") int limit
    );
}
