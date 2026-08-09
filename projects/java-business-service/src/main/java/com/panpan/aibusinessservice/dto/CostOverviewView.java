package com.panpan.aibusinessservice.dto;

import com.panpan.aibusinessservice.mapper.SummaryRow;
import java.util.List;

public record CostOverviewView(
        List<SummaryRow> byModel,
        List<SummaryRow> byIntent,
        SummaryRow totals
) {
}
