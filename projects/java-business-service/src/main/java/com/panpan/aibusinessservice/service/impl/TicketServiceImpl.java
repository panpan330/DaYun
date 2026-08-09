package com.panpan.aibusinessservice.service.impl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.panpan.aibusinessservice.common.cache.TicketIdempotencyCache;
import com.panpan.aibusinessservice.common.cache.TicketIdempotencyCacheEntry;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.CreateTicketCommand;
import com.panpan.aibusinessservice.dto.TicketToolView;
import com.panpan.aibusinessservice.entity.Order;
import com.panpan.aibusinessservice.entity.Ticket;
import com.panpan.aibusinessservice.entity.TicketStatus;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.OrderMapper;
import com.panpan.aibusinessservice.mapper.TicketMapper;
import com.panpan.aibusinessservice.service.TicketService;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Pattern;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class TicketServiceImpl implements TicketService {
    private static final Pattern IDEMPOTENCY_KEY_PATTERN = Pattern.compile("^[A-Za-z0-9._:-]{8,128}$");

    private final OrderMapper orderMapper;
    private final TicketMapper ticketMapper;
    private final TicketIdempotencyCache ticketIdempotencyCache;
    private final ObjectMapper objectMapper;

    public TicketServiceImpl(
            OrderMapper orderMapper,
            TicketMapper ticketMapper,
            TicketIdempotencyCache ticketIdempotencyCache,
            ObjectMapper objectMapper
    ) {
        this.orderMapper = orderMapper;
        this.ticketMapper = ticketMapper;
        this.ticketIdempotencyCache = ticketIdempotencyCache;
        this.objectMapper = objectMapper;
    }

    @Override
    @Transactional
    public TicketToolView createTicket(
            CreateTicketCommand command,
            InternalRequestContext context,
            String idempotencyKey
    ) {
        String normalizedKey = validateIdempotencyKey(idempotencyKey);
        validateRelatedOrder(command, context);
        return TicketToolView.from(createIdempotently(command, context, normalizedKey));
    }

    private Ticket createIdempotently(
            CreateTicketCommand command,
            InternalRequestContext context,
            String idempotencyKey
    ) {
        String fingerprint = TicketRequestFingerprint.from(command, context.userId(), context.tenantId());
        Optional<Ticket> cachedTicket = findFromIdempotencyCache(context.tenantId(), idempotencyKey, fingerprint);
        if (cachedTicket.isPresent()) {
            return cachedTicket.get();
        }

        Optional<Ticket> existingTicket = findByIdempotencyKey(context.tenantId(), idempotencyKey);
        if (existingTicket.isPresent()) {
            return existingTicketOrConflict(existingTicket.get(), fingerprint);
        }

        Ticket ticket = new Ticket(
                "T-" + UUID.randomUUID(),
                context.userId(),
                context.tenantId(),
                TicketStatus.CREATED.code(),
                command.title(),
                command.description(),
                command.category().code(),
                command.priority().code(),
                command.relatedOrderId(),
                command.source(),
                command.confirmationId(),
                idempotencyKey,
                fingerprint,
                context.traceId(),
                Instant.now()
        );

        try {
            ticketMapper.insertTicket(ticket);
            insertCreatedEvent(ticket);
            rememberIdempotency(ticket);
            return ticket;
        } catch (DuplicateKeyException exception) {
            return findByIdempotencyKey(context.tenantId(), idempotencyKey)
                    .map(record -> existingTicketOrConflict(record, fingerprint))
                    .orElseThrow(() -> exception);
        }
    }

    private String validateIdempotencyKey(String idempotencyKey) {
        if (idempotencyKey == null || idempotencyKey.isBlank()) {
            throw new BusinessException(BusinessErrorCode.IDEMPOTENCY_KEY_REQUIRED);
        }
        String normalizedKey = idempotencyKey.trim();
        if (!IDEMPOTENCY_KEY_PATTERN.matcher(normalizedKey).matches()) {
            throw new BusinessException(BusinessErrorCode.IDEMPOTENCY_KEY_INVALID);
        }
        return normalizedKey;
    }

    private void validateRelatedOrder(CreateTicketCommand command, InternalRequestContext context) {
        if (command.relatedOrderId() == null || command.relatedOrderId().isBlank()) {
            return;
        }

        Order order = Optional.ofNullable(
                        orderMapper.selectByTenantIdAndOrderId(context.tenantId(), command.relatedOrderId())
                )
                .orElseThrow(() -> new BusinessException(BusinessErrorCode.ORDER_NOT_FOUND));

        if (!order.visibleTo(context.userId(), context.tenantId())) {
            throw new BusinessException(BusinessErrorCode.ORDER_ACCESS_DENIED);
        }

        if (!order.isCanCreateTicket()) {
            throw new BusinessException(BusinessErrorCode.ORDER_NOT_SUPPORT_TICKET);
        }
    }

    private Optional<Ticket> findByIdempotencyKey(String tenantId, String idempotencyKey) {
        return Optional.ofNullable(ticketMapper.selectByTenantIdAndIdempotencyKey(tenantId, idempotencyKey));
    }

    private Optional<Ticket> findByTenantIdAndTicketId(String tenantId, String ticketId) {
        return Optional.ofNullable(ticketMapper.selectByTenantIdAndTicketId(tenantId, ticketId));
    }

    private Optional<Ticket> findFromIdempotencyCache(String tenantId, String idempotencyKey, String fingerprint) {
        Optional<TicketIdempotencyCacheEntry> cached = ticketIdempotencyCache.get(tenantId, idempotencyKey);
        if (cached.isEmpty()) {
            return Optional.empty();
        }
        TicketIdempotencyCacheEntry entry = cached.get();
        if (!entry.requestFingerprint().equals(fingerprint)) {
            if (findByTenantIdAndTicketId(tenantId, entry.ticketId()).isPresent()) {
                throw new BusinessException(BusinessErrorCode.IDEMPOTENCY_KEY_CONFLICT);
            }
            return Optional.empty();
        }
        return findByTenantIdAndTicketId(tenantId, entry.ticketId());
    }

    private Ticket existingTicketOrConflict(Ticket ticket, String fingerprint) {
        if (!ticket.getRequestFingerprint().equals(fingerprint)) {
            throw new BusinessException(BusinessErrorCode.IDEMPOTENCY_KEY_CONFLICT);
        }
        rememberIdempotency(ticket);
        return ticket;
    }

    private void rememberIdempotency(Ticket ticket) {
        ticketIdempotencyCache.put(
                ticket.getTenantId(),
                ticket.getIdempotencyKey(),
                new TicketIdempotencyCacheEntry(ticket.getRequestFingerprint(), ticket.getTicketId())
        );
    }

    private void insertCreatedEvent(Ticket ticket) {
        ticketMapper.insertTicketEvent(
                ticket.getTenantId(),
                "E-" + UUID.randomUUID(),
                ticket.getTicketId(),
                "created",
                eventPayload(ticket),
                ticket.getSource(),
                ticket.getRequesterUserId(),
                ticket.getCreatedTraceId(),
                ticket.getCreatedAt()
        );
    }

    private String eventPayload(Ticket ticket) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("ticket_id", ticket.getTicketId());
        payload.put("related_order_id", ticket.getRelatedOrderId());
        payload.put("category", ticket.getCategory());
        payload.put("priority", ticket.getPriority());
        payload.put("confirmation_id", ticket.getConfirmationId());
        payload.put("idempotency_key", ticket.getIdempotencyKey());
        try {
            return objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to serialize ticket event payload", exception);
        }
    }
}
