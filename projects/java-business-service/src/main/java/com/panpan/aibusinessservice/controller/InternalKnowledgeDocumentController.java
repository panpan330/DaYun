package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.common.trace.TraceHeaders;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentView;
import com.panpan.aibusinessservice.dto.KnowledgeDocumentWriteRequest;
import com.panpan.aibusinessservice.service.KnowledgeDocumentService;
import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/knowledge-documents")
public class InternalKnowledgeDocumentController {
    private final InternalRequestResolver requestResolver;
    private final KnowledgeDocumentService knowledgeDocumentService;

    public InternalKnowledgeDocumentController(
            InternalRequestResolver requestResolver,
            KnowledgeDocumentService knowledgeDocumentService) {
        this.requestResolver = requestResolver;
        this.knowledgeDocumentService = knowledgeDocumentService;
    }

    @GetMapping
    public ApiResponse<List<KnowledgeDocumentView>> list(
            HttpServletRequest request) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                knowledgeDocumentService.listAllDocuments(context.tenantId()),
                context.traceId());
    }

    @PostMapping
    public ApiResponse<KnowledgeDocumentView> upsert(
            @RequestBody KnowledgeDocumentWriteRequest body,
            @RequestHeader(value = TraceHeaders.IDEMPOTENCY_KEY, required = false) String idempotencyKey,
            HttpServletRequest request) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                knowledgeDocumentService.upsertDocument(context, body, idempotencyKey),
                context.traceId());
    }

    @DeleteMapping("/{documentId}")
    public ApiResponse<Boolean> delete(
            @PathVariable String documentId,
            HttpServletRequest request) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                knowledgeDocumentService.deleteDocument(context, documentId),
                context.traceId());
    }
}
