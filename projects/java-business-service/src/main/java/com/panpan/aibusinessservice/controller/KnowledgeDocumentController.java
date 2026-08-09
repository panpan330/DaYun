package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import com.panpan.aibusinessservice.dto.CurrentUserView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentView;
import com.panpan.aibusinessservice.service.AuthService;
import com.panpan.aibusinessservice.service.KnowledgeDocumentService;
import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/knowledge-documents")
public class KnowledgeDocumentController {
    private final AuthService authService;
    private final KnowledgeDocumentService knowledgeDocumentService;

    public KnowledgeDocumentController(
            AuthService authService,
            KnowledgeDocumentService knowledgeDocumentService
    ) {
        this.authService = authService;
        this.knowledgeDocumentService = knowledgeDocumentService;
    }

    @GetMapping
    public ApiResponse<List<KnowledgeDocumentView>> listDocuments(
            @RequestHeader(value = "Authorization", required = false) String authorization,
            HttpServletRequest servletRequest
    ) {
        CurrentUserView currentUser = authService.currentUser(authorization);
        return ApiResponse.ok(
                knowledgeDocumentService.listVisibleDocuments(currentUser),
                TraceFilter.currentTraceId(servletRequest)
        );
    }
}
