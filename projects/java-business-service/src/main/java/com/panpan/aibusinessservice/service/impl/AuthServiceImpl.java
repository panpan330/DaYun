package com.panpan.aibusinessservice.service.impl;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.LoginRequest;
import com.panpan.aibusinessservice.dto.LoginResponse;
import com.panpan.aibusinessservice.entity.User;
import com.panpan.aibusinessservice.exception.BusinessErrorCode;
import com.panpan.aibusinessservice.exception.BusinessException;
import com.panpan.aibusinessservice.mapper.UserMapper;
import com.panpan.aibusinessservice.service.AuthService;
import java.util.List;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

@Service
public class AuthServiceImpl implements AuthService {
    private static final String LOCAL_DEV_TOKEN_PREFIX = "local-dev-token:";
    private static final String BEARER_PREFIX = "Bearer ";
    private static final Pattern SAFE_ID_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{1,64}$");

    private final UserMapper userMapper;

    public AuthServiceImpl(UserMapper userMapper) {
        this.userMapper = userMapper;
    }

    @Override
    public LoginResponse login(LoginRequest request) {
        String tenantId = normalizeSafeId(request.normalizedTenantId(), BusinessErrorCode.LOGIN_REQUEST_INVALID);
        String username = normalizeSafeId(request.username(), BusinessErrorCode.LOGIN_REQUEST_INVALID);

        User user = userMapper.selectActiveByTenantIdAndUsername(tenantId, username);
        if (user == null || !matchesPassword(request.password(), user.getPasswordHash())) {
            throw new BusinessException(BusinessErrorCode.LOGIN_FAILED);
        }

        List<String> roles = userMapper.selectRoleCodesByTenantIdAndUserId(user.getTenantId(), user.getUserId());
        CurrentUserView currentUser = CurrentUserView.from(user, roles);
        return new LoginResponse(buildLocalDevToken(user), currentUser);
    }

    @Override
    public CurrentUserView currentUser(String authorizationHeader) {
        TokenSubject subject = parseLocalDevToken(authorizationHeader);
        User user = userMapper.selectActiveByTenantIdAndUserId(subject.tenantId(), subject.userId());
        if (user == null) {
            throw new BusinessException(BusinessErrorCode.USER_NOT_FOUND);
        }
        List<String> roles = userMapper.selectRoleCodesByTenantIdAndUserId(user.getTenantId(), user.getUserId());
        return CurrentUserView.from(user, roles);
    }

    private String buildLocalDevToken(User user) {
        return LOCAL_DEV_TOKEN_PREFIX + user.getTenantId() + ":" + user.getUserId();
    }

    private TokenSubject parseLocalDevToken(String authorizationHeader) {
        if (authorizationHeader == null || !authorizationHeader.startsWith(BEARER_PREFIX)) {
            throw new BusinessException(BusinessErrorCode.AUTH_REQUIRED);
        }
        String token = authorizationHeader.substring(BEARER_PREFIX.length()).trim();
        if (!token.startsWith(LOCAL_DEV_TOKEN_PREFIX)) {
            throw new BusinessException(BusinessErrorCode.AUTH_REQUIRED);
        }
        String payload = token.substring(LOCAL_DEV_TOKEN_PREFIX.length());
        String[] parts = payload.split(":", 2);
        if (parts.length != 2) {
            throw new BusinessException(BusinessErrorCode.AUTH_REQUIRED);
        }
        return new TokenSubject(
                normalizeSafeId(parts[0], BusinessErrorCode.AUTH_REQUIRED),
                normalizeSafeId(parts[1], BusinessErrorCode.AUTH_REQUIRED)
        );
    }

    private boolean matchesPassword(String rawPassword, String storedPasswordHash) {
        if (rawPassword == null || storedPasswordHash == null) {
            return false;
        }
        if (storedPasswordHash.startsWith("{plain}")) {
            return storedPasswordHash.substring("{plain}".length()).equals(rawPassword);
        }
        return false;
    }

    private String normalizeSafeId(String value, BusinessErrorCode errorCode) {
        if (value == null) {
            throw new BusinessException(errorCode);
        }
        String trimmed = value.trim();
        if (!SAFE_ID_PATTERN.matcher(trimmed).matches()) {
            throw new BusinessException(errorCode);
        }
        return trimmed;
    }

    private record TokenSubject(String tenantId, String userId) {
    }
}
