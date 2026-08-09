package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.OrderListItemView;
import java.util.List;

public interface OrderQueryService {
    List<OrderListItemView> listVisibleOrders(CurrentUserView currentUser);
}
