package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.User;
import java.util.List;
import org.apache.ibatis.annotations.Param;

public interface UserMapper {
    User selectActiveByTenantIdAndUsername(
            @Param("tenantId") String tenantId,
            @Param("username") String username
    );

    User selectActiveByTenantIdAndUserId(
            @Param("tenantId") String tenantId,
            @Param("userId") String userId
    );

    List<String> selectRoleCodesByTenantIdAndUserId(
            @Param("tenantId") String tenantId,
            @Param("userId") String userId
    );

    List<User> selectActiveStaffByTenantId(@Param("tenantId") String tenantId);
}
