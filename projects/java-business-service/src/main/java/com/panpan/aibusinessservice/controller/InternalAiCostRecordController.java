package com.panpan.aibusinessservice.controller;

import com.panpan.aibusinessservice.common.ApiResponse;
import com.panpan.aibusinessservice.common.security.InternalRequestContext;
import com.panpan.aibusinessservice.common.security.InternalRequestResolver;
import com.panpan.aibusinessservice.dto.CostOverviewView;
import com.panpan.aibusinessservice.dto.CostRecordCommand;
import com.panpan.aibusinessservice.entity.AiCostRecord;
import com.panpan.aibusinessservice.mapper.AiCostRecordMapper;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/internal/ai-cost-records")
public class InternalAiCostRecordController {

    private final InternalRequestResolver requestResolver;
    private final AiCostRecordMapper costRecordMapper;

    public InternalAiCostRecordController(
            InternalRequestResolver requestResolver,
            AiCostRecordMapper costRecordMapper
    ) {
        this.requestResolver = requestResolver;
        this.costRecordMapper = costRecordMapper;
    }

    @PostMapping
    public ApiResponse<Void> batchUpsert(
            @Valid @RequestBody CostRecordCommand command,
            HttpServletRequest request
    ) {
        InternalRequestContext context = requestResolver.resolve(request);
        List<AiCostRecord> records = command.records().stream().map(item -> {
            AiCostRecord record = new AiCostRecord();
            record.setModel(item.model());
            record.setIntent(item.intent());
            record.setCallCount(item.callCount());
            record.setInputTokens(item.inputTokens());
            record.setOutputTokens(item.outputTokens());
            record.setTotalTokens(item.totalTokens());
            record.setEstimatedCost(item.estimatedCost());
            record.setWindowStart(item.windowStart());
            record.setWindowEnd(item.windowEnd());
            return record;
        }).toList();
        if (!records.isEmpty()) {
            costRecordMapper.batchUpsert(records);
        }
        return ApiResponse.ok(null, context.traceId());
    }

    @GetMapping("/overview")
    public ApiResponse<CostOverviewView> overview(HttpServletRequest request) {
        InternalRequestContext context = requestResolver.resolve(request);
        return ApiResponse.ok(
                new CostOverviewView(
                        costRecordMapper.summarizeByModel(),
                        costRecordMapper.summarizeByIntent(),
                        costRecordMapper.summarizeTotals()),
                context.traceId());
    }
}
