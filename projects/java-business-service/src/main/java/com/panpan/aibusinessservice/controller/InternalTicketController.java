package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.service.TicketService;
import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.dto.CreateTicketCommand;
import com.panpan.aibusinessservice.dto.TicketToolView;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/tickets")
public class InternalTicketController {
    private final InternalRequestResolver requestResolver;
    private final TicketService ticketService;

    public InternalTicketController(
            InternalRequestResolver requestResolver,
            TicketService ticketService
    ) {
        this.requestResolver = requestResolver;
        this.ticketService = ticketService;
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public ApiResponse<TicketToolView> createTicket(
            @Valid @RequestBody CreateTicketCommand command,
            @RequestHeader(value = TraceHeaders.IDEMPOTENCY_KEY, required = false) String idempotencyKey,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        TicketToolView ticket = ticketService.createTicket(command, context, idempotencyKey);
        return ApiResponse.ok(ticket, context.traceId());
    }
}
