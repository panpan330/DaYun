package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.StaffUserView;
import com.panpan.aibusinessservice.entity.User;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.UserMapper;
import com.panpan.aibusinessservice.service.UserDirectoryService;
import java.util.List;
import java.util.Set;
import org.springframework.stereotype.Service;

@Service
public class UserDirectoryServiceImpl implements UserDirectoryService {
    private static final Set<String> STAFF_ROLES = Set.of("agent", "supervisor", "admin");

    private final UserMapper userMapper;

    public UserDirectoryServiceImpl(UserMapper userMapper) {
        this.userMapper = userMapper;
    }

    @Override
    public List<StaffUserView> listAssignableStaff(CurrentUserView currentUser) {
        requireStaff(currentUser);
        return userMapper.selectActiveStaffByTenantId(currentUser.tenantId())
                .stream()
                .map(this::toView)
                .toList();
    }

    private StaffUserView toView(User user) {
        List<String> roles = userMapper.selectRoleCodesByTenantIdAndUserId(user.getTenantId(), user.getUserId());
        return StaffUserView.from(user, roles);
    }

    private void requireStaff(CurrentUserView currentUser) {
        boolean staff = currentUser.roles().stream().anyMatch(STAFF_ROLES::contains);
        if (!staff) {
            throw new BusinessException(BusinessErrorCode.TICKET_ACCESS_DENIED);
        }
    }
}
