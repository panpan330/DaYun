package com.panpan.aibusinessservice.service.impl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.dto.AssignTicketRequest;
import com.panpan.aibusinessservice.dto.AddTicketMessageRequest;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.ResolveTicketRequest;
import com.panpan.aibusinessservice.dto.ReopenTicketRequest;
import com.panpan.aibusinessservice.dto.TicketDetailView;
import com.panpan.aibusinessservice.dto.TicketEventView;
import com.panpan.aibusinessservice.dto.TicketListItemView;
import com.panpan.aibusinessservice.dto.TicketMessageView;
import com.panpan.aibusinessservice.dto.UpdateTicketStatusRequest;
import com.panpan.aibusinessservice.entity.Ticket;
import com.panpan.aibusinessservice.entity.User;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.TicketMapper;
import com.panpan.aibusinessservice.mapper.UserMapper;
import com.panpan.aibusinessservice.service.TicketQueryService;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class TicketQueryServiceImpl implements TicketQueryService {
    private static final Set<String> STAFF_ROLES = Set.of("agent", "supervisor", "admin");
    private static final Set<String> MANAGER_ROLES = Set.of("supervisor", "admin");
    private static final Set<String> CUSTOMER_REPLY_ALLOWED_STATUSES = Set.of("created", "in_progress", "waiting_user");
    private static final Map<String, Set<String>> ALLOWED_TRANSITIONS = Map.of(
            "created", Set.of("in_progress", "waiting_user"),
            "in_progress", Set.of("waiting_user", "resolved"),
            "waiting_user", Set.of("in_progress", "resolved"),
            "resolved", Set.of("closed"),
            "closed", Set.of()
    );

    private final TicketMapper ticketMapper;
    private final UserMapper userMapper;
    private final ObjectMapper objectMapper;

    public TicketQueryServiceImpl(TicketMapper ticketMapper, UserMapper userMapper, ObjectMapper objectMapper) {
        this.ticketMapper = ticketMapper;
        this.userMapper = userMapper;
        this.objectMapper = objectMapper;
    }

    @Override
    public List<TicketListItemView> listVisibleTickets(CurrentUserView currentUser) {
        if (currentUser.roles().contains("customer")) {
            return ticketMapper.selectByTenantIdAndRequesterUserId(currentUser.tenantId(), currentUser.userId())
                    .stream()
                    .map(TicketListItemView::from)
                    .toList();
        }

        return ticketMapper.selectAllByTenantId(currentUser.tenantId())
                .stream()
                .map(TicketListItemView::from)
                .toList();
    }

    @Override
    public TicketDetailView getVisibleTicket(CurrentUserView currentUser, String ticketId) {
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        return toDetail(ticket, canViewInternal(ticket, currentUser));
    }

    @Override
    @Transactional
    public TicketDetailView updateTicketStatus(
            CurrentUserView currentUser,
            String ticketId,
            UpdateTicketStatusRequest request,
            String traceId
    ) {
        requireStaff(currentUser);
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        validateTransition(ticket.getTicketStatus(), request.targetStatus());

        Instant now = Instant.now();
        ticketMapper.updateTicketStatus(
                ticket.getTenantId(),
                ticket.getTicketId(),
                request.targetStatus(),
                now
        );
        ticket.setTicketStatus(request.targetStatus());
        ticket.setUpdatedAt(now);
        insertStatusChangedEvent(ticket, currentUser, request, traceId, now);
        return toDetail(ticket, true);
    }

    @Override
    @Transactional
    public TicketDetailView claimTicket(CurrentUserView currentUser, String ticketId, String traceId) {
        requireStaff(currentUser);
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        if (ticket.getAssigneeUserId() != null && !ticket.getAssigneeUserId().isBlank()) {
            throw new BusinessException(BusinessErrorCode.TICKET_ALREADY_ASSIGNED);
        }

        User assignee = requireAssignableStaff(currentUser.tenantId(), currentUser.userId());
        Instant now = Instant.now();
        upsertAssignment(ticket, assignee, currentUser, now);
        ticket.setAssigneeUserId(assignee.getUserId());
        ticket.setAssigneeDisplayName(assignee.getDisplayName());
        insertAssignmentEvent(ticket, currentUser, "ticket_claimed", assignee, null, null, traceId, now);
        return toDetail(ticket, true);
    }

    @Override
    @Transactional
    public TicketDetailView assignTicket(
            CurrentUserView currentUser,
            String ticketId,
            AssignTicketRequest request,
            String traceId
    ) {
        requireStaff(currentUser);
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);

        User assignee = requireAssignableStaff(currentUser.tenantId(), request.assigneeUserId().trim());
        requireAssignmentPermission(currentUser, ticket, assignee.getUserId());
        String previousAssigneeUserId = ticket.getAssigneeUserId();
        Instant now = Instant.now();
        upsertAssignment(ticket, assignee, currentUser, now);
        ticket.setAssigneeUserId(assignee.getUserId());
        ticket.setAssigneeDisplayName(assignee.getDisplayName());
        insertAssignmentEvent(
                ticket,
                currentUser,
                previousAssigneeUserId == null || previousAssigneeUserId.isBlank()
                        ? "ticket_assigned"
                        : "ticket_transferred",
                assignee,
                previousAssigneeUserId,
                normalizeNote(request.note()),
                traceId,
                now
        );
        return toDetail(ticket, true);
    }

    @Override
    @Transactional
    public TicketDetailView addTicketMessage(
            CurrentUserView currentUser,
            String ticketId,
            AddTicketMessageRequest request,
            String traceId
    ) {
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        if (currentUser.roles().contains("customer")) {
            return addCustomerMessage(ticket, currentUser, request, traceId);
        }

        requireStaff(currentUser);
        Instant now = Instant.now();
        insertTicketMessage(ticket, request.visibility(), request.content(), "staff", currentUser, traceId, now);
        ticketMapper.touchTicketUpdatedAt(ticket.getTenantId(), ticket.getTicketId(), now);
        ticket.setUpdatedAt(now);
        return toDetail(ticket, true);
    }

    @Override
    @Transactional
    public TicketDetailView resolveTicket(
            CurrentUserView currentUser,
            String ticketId,
            ResolveTicketRequest request,
            String traceId
    ) {
        requireStaff(currentUser);
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        validateTransition(ticket.getTicketStatus(), "resolved");

        Instant now = Instant.now();
        insertTicketMessage(ticket, "public", request.content(), "staff", currentUser, traceId, now);
        ticketMapper.updateTicketStatus(ticket.getTenantId(), ticket.getTicketId(), "resolved", now);
        ticket.setTicketStatus("resolved");
        ticket.setUpdatedAt(now);
        insertTicketLifecycleEvent(
                ticket,
                "ticket_resolved",
                Map.of("target_status", "resolved"),
                "staff",
                currentUser.userId(),
                traceId,
                now
        );
        return toDetail(ticket, true);
    }

    @Override
    @Transactional
    public TicketDetailView reopenTicket(
            CurrentUserView currentUser,
            String ticketId,
            ReopenTicketRequest request,
            String traceId
    ) {
        requireCustomer(currentUser);
        Ticket ticket = requireVisibleTicket(currentUser, ticketId);
        if (!"resolved".equals(ticket.getTicketStatus())) {
            throw new BusinessException(BusinessErrorCode.TICKET_REOPEN_NOT_ALLOWED);
        }

        Instant now = Instant.now();
        insertTicketMessage(ticket, "public", request.content(), "customer", currentUser, traceId, now);
        ticketMapper.updateTicketStatus(ticket.getTenantId(), ticket.getTicketId(), "in_progress", now);
        ticket.setTicketStatus("in_progress");
        ticket.setUpdatedAt(now);
        insertTicketLifecycleEvent(
                ticket,
                "ticket_reopened",
                Map.of("previous_status", "resolved", "target_status", "in_progress"),
                "customer",
                currentUser.userId(),
                traceId,
                now
        );
        return toDetail(ticket, false);
    }

    private TicketDetailView addCustomerMessage(
            Ticket ticket,
            CurrentUserView currentUser,
            AddTicketMessageRequest request,
            String traceId
    ) {
        if (!"public".equals(request.visibility())) {
            throw new BusinessException(BusinessErrorCode.TICKET_MESSAGE_VISIBILITY_INVALID);
        }
        if (!CUSTOMER_REPLY_ALLOWED_STATUSES.contains(ticket.getTicketStatus())) {
            throw new BusinessException(BusinessErrorCode.TICKET_CUSTOMER_REPLY_NOT_ALLOWED);
        }

        Instant now = Instant.now();
        insertTicketMessage(ticket, "public", request.content(), "customer", currentUser, traceId, now);
        if ("waiting_user".equals(ticket.getTicketStatus())) {
            ticketMapper.updateTicketStatus(ticket.getTenantId(), ticket.getTicketId(), "in_progress", now);
            ticket.setTicketStatus("in_progress");
            insertCustomerReplyResumedEvent(ticket, currentUser, traceId, now);
        } else {
            ticketMapper.touchTicketUpdatedAt(ticket.getTenantId(), ticket.getTicketId(), now);
        }
        ticket.setUpdatedAt(now);
        return toDetail(ticket, false);
    }

    private void insertTicketMessage(
            Ticket ticket,
            String visibility,
            String content,
            String authorType,
            CurrentUserView currentUser,
            String traceId,
            Instant createdAt
    ) {
        ticketMapper.insertTicketMessage(
                ticket.getTenantId(),
                "M-" + UUID.randomUUID(),
                ticket.getTicketId(),
                visibility,
                content.trim(),
                authorType,
                currentUser.userId(),
                currentUser.displayName(),
                traceId,
                createdAt
        );
    }

    private Ticket requireVisibleTicket(CurrentUserView currentUser, String ticketId) {
        Ticket ticket = ticketMapper.selectByTenantIdAndTicketId(currentUser.tenantId(), ticketId);
        if (ticket == null) {
            throw new BusinessException(BusinessErrorCode.TICKET_NOT_FOUND);
        }
        if (currentUser.roles().contains("customer")
                && !ticket.getRequesterUserId().equals(currentUser.userId())) {
            throw new BusinessException(BusinessErrorCode.TICKET_ACCESS_DENIED);
        }
        return ticket;
    }

    private void requireStaff(CurrentUserView currentUser) {
        boolean staff = currentUser.roles().stream().anyMatch(STAFF_ROLES::contains);
        if (!staff) {
            throw new BusinessException(BusinessErrorCode.TICKET_ACCESS_DENIED);
        }
    }

    private void requireCustomer(CurrentUserView currentUser) {
        if (!currentUser.roles().contains("customer")) {
            throw new BusinessException(BusinessErrorCode.TICKET_ACCESS_DENIED);
        }
    }

    private void requireAssignmentPermission(CurrentUserView currentUser, Ticket ticket, String targetAssigneeUserId) {
        boolean manager = currentUser.roles().stream().anyMatch(MANAGER_ROLES::contains);
        boolean currentAssignee = currentUser.userId().equals(ticket.getAssigneeUserId());
        boolean unassigned = ticket.getAssigneeUserId() == null || ticket.getAssigneeUserId().isBlank();
        boolean selfClaimThroughAssignEndpoint = unassigned && currentUser.userId().equals(targetAssigneeUserId);
        if (!manager && !currentAssignee && !selfClaimThroughAssignEndpoint) {
            throw new BusinessException(BusinessErrorCode.TICKET_ACCESS_DENIED);
        }
    }

    private User requireAssignableStaff(String tenantId, String userId) {
        User user = userMapper.selectActiveByTenantIdAndUserId(tenantId, userId);
        if (user == null) {
            throw new BusinessException(BusinessErrorCode.TICKET_ASSIGNEE_INVALID);
        }
        List<String> roles = userMapper.selectRoleCodesByTenantIdAndUserId(tenantId, userId);
        boolean staff = roles.stream().anyMatch(STAFF_ROLES::contains);
        if (!staff) {
            throw new BusinessException(BusinessErrorCode.TICKET_ASSIGNEE_INVALID);
        }
        return user;
    }

    private void upsertAssignment(Ticket ticket, User assignee, CurrentUserView currentUser, Instant assignedAt) {
        int updatedRows = ticketMapper.updateTicketAssignment(
                ticket.getTenantId(),
                ticket.getTicketId(),
                assignee.getUserId(),
                assignee.getDisplayName(),
                currentUser.userId(),
                assignedAt
        );
        if (updatedRows == 0) {
            ticketMapper.insertTicketAssignment(
                    ticket.getTenantId(),
                    ticket.getTicketId(),
                    assignee.getUserId(),
                    assignee.getDisplayName(),
                    currentUser.userId(),
                    assignedAt
            );
        }
    }

    private void validateTransition(String currentStatus, String targetStatus) {
        Set<String> allowedTargets = ALLOWED_TRANSITIONS.getOrDefault(currentStatus, Set.of());
        if (!allowedTargets.contains(targetStatus)) {
            throw new BusinessException(BusinessErrorCode.TICKET_STATUS_TRANSITION_INVALID);
        }
    }

    private boolean canViewInternal(Ticket ticket, CurrentUserView currentUser) {
        return !currentUser.roles().contains("customer")
                || !ticket.getRequesterUserId().equals(currentUser.userId());
    }

    private TicketDetailView toDetail(Ticket ticket, boolean includeInternal) {
        List<TicketEventView> events = ticketMapper
                .selectEventsByTenantIdAndTicketId(ticket.getTenantId(), ticket.getTicketId())
                .stream()
                .map(event -> toEventView(event, includeInternal))
                .toList();
        List<TicketMessageView> messages = ticketMapper
                .selectMessagesByTenantIdAndTicketId(ticket.getTenantId(), ticket.getTicketId(), includeInternal)
                .stream()
                .map(TicketMessageView::from)
                .toList();
        return TicketDetailView.from(ticket, events, messages);
    }

    private TicketEventView toEventView(com.panpan.aibusinessservice.entity.TicketEvent event, boolean includeInternal) {
        if (includeInternal || event.getEventPayload() == null || event.getEventPayload().isBlank()) {
            return TicketEventView.from(event);
        }
        try {
            Map<String, Object> payload = objectMapper.readValue(event.getEventPayload(), LinkedHashMap.class);
            payload.remove("note");
            return new TicketEventView(
                    event.getEventId(),
                    event.getEventType(),
                    objectMapper.writeValueAsString(payload),
                    event.getOperatorType(),
                    event.getOperatorId(),
                    event.getTraceId(),
                    event.getCreatedAt()
            );
        } catch (JsonProcessingException exception) {
            return TicketEventView.from(event);
        }
    }

    private void insertStatusChangedEvent(
            Ticket ticket,
            CurrentUserView currentUser,
            UpdateTicketStatusRequest request,
            String traceId,
            Instant createdAt
    ) {
        ticketMapper.insertTicketEvent(
                ticket.getTenantId(),
                "E-" + UUID.randomUUID(),
                ticket.getTicketId(),
                "status_changed",
                statusChangedPayload(request),
                "staff",
                currentUser.userId(),
                traceId,
                createdAt
        );
    }

    private String statusChangedPayload(UpdateTicketStatusRequest request) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("target_status", request.targetStatus());
        payload.put("note", request.note());
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to serialize ticket status event payload", exception);
        }
    }

    private void insertCustomerReplyResumedEvent(
            Ticket ticket,
            CurrentUserView currentUser,
            String traceId,
            Instant createdAt
    ) {
        insertTicketLifecycleEvent(
                ticket,
                "status_changed",
                Map.of("target_status", "in_progress", "reason", "customer_replied"),
                "customer",
                currentUser.userId(),
                traceId,
                createdAt
        );
    }

    private void insertTicketLifecycleEvent(
            Ticket ticket,
            String eventType,
            Map<String, ?> payload,
            String operatorType,
            String operatorId,
            String traceId,
            Instant createdAt
    ) {
        try {
            ticketMapper.insertTicketEvent(
                    ticket.getTenantId(),
                    "E-" + UUID.randomUUID(),
                    ticket.getTicketId(),
                    eventType,
                    objectMapper.writeValueAsString(payload),
                    operatorType,
                    operatorId,
                    traceId,
                    createdAt
            );
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to serialize ticket lifecycle event payload", exception);
        }
    }

    private void insertAssignmentEvent(
            Ticket ticket,
            CurrentUserView currentUser,
            String eventType,
            User assignee,
            String previousAssigneeUserId,
            String note,
            String traceId,
            Instant createdAt
    ) {
        ticketMapper.insertTicketEvent(
                ticket.getTenantId(),
                "E-" + UUID.randomUUID(),
                ticket.getTicketId(),
                eventType,
                assignmentPayload(assignee, previousAssigneeUserId, note),
                "staff",
                currentUser.userId(),
                traceId,
                createdAt
        );
    }

    private String assignmentPayload(User assignee, String previousAssigneeUserId, String note) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("assignee_user_id", assignee.getUserId());
        payload.put("assignee_display_name", assignee.getDisplayName());
        payload.put("previous_assignee_user_id", previousAssigneeUserId);
        payload.put("note", note);
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to serialize ticket assignment event payload", exception);
        }
    }

    private String normalizeNote(String note) {
        if (note == null || note.isBlank()) {
            return null;
        }
        return note.trim();
    }
}
