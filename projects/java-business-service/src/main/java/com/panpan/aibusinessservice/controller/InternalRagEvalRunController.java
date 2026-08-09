package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.SaveRagEvalRunCommand;
import com.panpan.aibusinessservice.service.RagEvalRunService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/rag-eval-runs")
public class InternalRagEvalRunController {
    private final InternalRequestResolver requestResolver;
    private final RagEvalRunService ragEvalRunService;

    public InternalRagEvalRunController(
            InternalRequestResolver requestResolver,
            RagEvalRunService ragEvalRunService
    ) {
        this.requestResolver = requestResolver;
        this.ragEvalRunService = ragEvalRunService;
    }

    @PostMapping
    public ApiResponse<Void> save(
            @Valid @RequestBody SaveRagEvalRunCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        ragEvalRunService.save(command);
        return ApiResponse.ok(null, context.traceId());
    }
}
