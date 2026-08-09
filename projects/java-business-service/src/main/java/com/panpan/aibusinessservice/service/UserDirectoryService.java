package com.panpan.aibusinessservice.service;

import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.StaffUserView;
import java.util.List;

public interface UserDirectoryService {
    List<StaffUserView> listAssignableStaff(CurrentUserView currentUser);
}
