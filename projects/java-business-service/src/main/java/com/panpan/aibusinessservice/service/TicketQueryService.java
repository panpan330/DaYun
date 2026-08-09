package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.ResolveTicketRequest;
import com.panpan.aibusinessservice.dto.ReopenTicketRequest;
import com.panpan.aibusinessservice.dto.AssignTicketRequest;
import com.panpan.aibusinessservice.dto.AddTicketMessageRequest;
import com.panpan.aibusinessservice.dto.TicketDetailView;
import com.panpan.aibusinessservice.dto.TicketListItemView;
import com.panpan.aibusinessservice.dto.UpdateTicketStatusRequest;
import java.util.List;

public interface TicketQueryService {
    List<TicketListItemView> listVisibleTickets(CurrentUserView currentUser);

    TicketDetailView getVisibleTicket(CurrentUserView currentUser, String ticketId);

    TicketDetailView updateTicketStatus(
            CurrentUserView currentUser,
            String ticketId,
            UpdateTicketStatusRequest request,
            String traceId
    );

    TicketDetailView claimTicket(CurrentUserView currentUser, String ticketId, String traceId);

    TicketDetailView assignTicket(
            CurrentUserView currentUser,
            String ticketId,
            AssignTicketRequest request,
            String traceId
    );

    TicketDetailView addTicketMessage(
            CurrentUserView currentUser,
            String ticketId,
            AddTicketMessageRequest request,
            String traceId
    );

    TicketDetailView resolveTicket(
            CurrentUserView currentUser,
            String ticketId,
            ResolveTicketRequest request,
            String traceId
    );

    TicketDetailView reopenTicket(
            CurrentUserView currentUser,
            String ticketId,
            ReopenTicketRequest request,
            String traceId
    );
}
