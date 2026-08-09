package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.Ticket;
import com.panpan.aibusinessservice.entity.TicketEvent;
import com.panpan.aibusinessservice.entity.TicketMessage;
import java.time.Instant;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface TicketMapper {
    Ticket selectByTenantIdAndIdempotencyKey(
            @Param("tenantId") String tenantId,
            @Param("idempotencyKey") String idempotencyKey
    );

    Ticket selectByTenantIdAndTicketId(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId
    );

    List<Ticket> selectByTenantIdAndRequesterUserId(
            @Param("tenantId") String tenantId,
            @Param("requesterUserId") String requesterUserId
    );

    List<Ticket> selectAllByTenantId(@Param("tenantId") String tenantId);

    int insertTicket(Ticket ticket);

    int insertTicketAssignment(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId,
            @Param("assigneeUserId") String assigneeUserId,
            @Param("assigneeDisplayName") String assigneeDisplayName,
            @Param("assignedByUserId") String assignedByUserId,
            @Param("assignedAt") Instant assignedAt
    );

    int updateTicketAssignment(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId,
            @Param("assigneeUserId") String assigneeUserId,
            @Param("assigneeDisplayName") String assigneeDisplayName,
            @Param("assignedByUserId") String assignedByUserId,
            @Param("assignedAt") Instant assignedAt
    );

    List<TicketEvent> selectEventsByTenantIdAndTicketId(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId
    );

    List<TicketMessage> selectMessagesByTenantIdAndTicketId(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId,
            @Param("includeInternal") boolean includeInternal
    );

    int updateTicketStatus(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId,
            @Param("ticketStatus") String ticketStatus,
            @Param("updatedAt") Instant updatedAt
    );

    int touchTicketUpdatedAt(
            @Param("tenantId") String tenantId,
            @Param("ticketId") String ticketId,
            @Param("updatedAt") Instant updatedAt
    );

    int insertTicketMessage(
            @Param("tenantId") String tenantId,
            @Param("messageId") String messageId,
            @Param("ticketId") String ticketId,
            @Param("visibility") String visibility,
            @Param("content") String content,
            @Param("authorType") String authorType,
            @Param("authorUserId") String authorUserId,
            @Param("authorDisplayName") String authorDisplayName,
            @Param("traceId") String traceId,
            @Param("createdAt") Instant createdAt
    );

    int insertTicketEvent(
            @Param("tenantId") String tenantId,
            @Param("eventId") String eventId,
            @Param("ticketId") String ticketId,
            @Param("eventType") String eventType,
            @Param("eventPayload") String eventPayload,
            @Param("operatorType") String operatorType,
            @Param("operatorId") String operatorId,
            @Param("traceId") String traceId,
            @Param("createdAt") java.time.Instant createdAt
    );
}
