package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.trace.TraceFilter;
import com.panpan.aibusinessservice.dto.RagEvalRunView;
import com.panpan.aibusinessservice.service.RagEvalRunService;
import jakarta.servlet.http.HttpServletRequest;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/rag-eval-runs")
public class RagEvalRunController {
    private final RagEvalRunService ragEvalRunService;

    public RagEvalRunController(RagEvalRunService ragEvalRunService) {
        this.ragEvalRunService = ragEvalRunService;
    }

    @GetMapping
    public ApiResponse<List<RagEvalRunView>> listRecent(
            @RequestParam(defaultValue = "20") int limit,
            HttpServletRequest request
    ) {
        return ApiResponse.ok(
                ragEvalRunService.listRecent(limit),
                TraceFilter.currentTraceId(request)
        );
    }

    @GetMapping("/latest")
    public ApiResponse<RagEvalRunView> latestByRetriever(
            @RequestParam(defaultValue = "keyword") String retriever,
            HttpServletRequest request
    ) {
        return ApiResponse.ok(
                ragEvalRunService.latestByRetriever(retriever),
                TraceFilter.currentTraceId(request)
        );
    }
}
