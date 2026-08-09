package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.dto.CreateTicketCommand;
import com.panpan.aibusinessservice.dto.TicketToolView;

public interface TicketService {
    TicketToolView createTicket(
            CreateTicketCommand command,
            InternalRequestContext context,
            String idempotencyKey
    );
}
