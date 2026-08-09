package com.panpan.aibusinessservice.mapper;

import com.panpan.aibusinessservice.entity.AiCostRecord;
import java.util.List;

public interface AiCostRecordMapper {
    int batchUpsert(List<AiCostRecord> records);
    List<SummaryRow> summarizeByModel();
    List<SummaryRow> summarizeByIntent();
    SummaryRow summarizeTotals();
}
