package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.LoginRequest;
import com.panpan.aibusinessservice.dto.LoginResponse;

public interface AuthService {
    LoginResponse login(LoginRequest request);

    CurrentUserView currentUser(String authorizationHeader);
}
