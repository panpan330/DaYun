package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.OrderListItemView;
import com.panpan.aibusinessservice.mapper.OrderMapper;
import com.panpan.aibusinessservice.service.OrderQueryService;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class OrderQueryServiceImpl implements OrderQueryService {
    private final OrderMapper orderMapper;

    public OrderQueryServiceImpl(OrderMapper orderMapper) {
        this.orderMapper = orderMapper;
    }

    @Override
    public List<OrderListItemView> listVisibleOrders(CurrentUserView currentUser) {
        if (currentUser.roles().contains("customer")) {
            return orderMapper.selectByTenantIdAndUserId(currentUser.tenantId(), currentUser.userId())
                    .stream()
                    .map(OrderListItemView::from)
                    .toList();
        }

        return orderMapper.selectAllByTenantId(currentUser.tenantId())
                .stream()
                .map(OrderListItemView::from)
                .toList();
    }
}
