package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.entity.AiCostRecord;
import com.panpan.aibusinessservice.mapper.AiCostRecordMapper;
import com.panpan.aibusinessservice.mapper.SummaryRow;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@SpringBootTest
class AiCostRecordMapperTest {

    @Autowired
    private AiCostRecordMapper mapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM ai_cost_records");
    }

    private AiCostRecord record(String model, String intent, int callCount, long input, long output, String cost) {
        Instant now = Instant.now();
        AiCostRecord r = new AiCostRecord();
        r.setModel(model);
        r.setIntent(intent);
        r.setCallCount(callCount);
        r.setInputTokens(input);
        r.setOutputTokens(output);
        r.setTotalTokens(input + output);
        r.setEstimatedCost(new BigDecimal(cost));
        r.setWindowStart(now);
        r.setWindowEnd(now);
        return r;
    }

    @Test
    void upsertMergesSameWindow() {
        mapper.batchUpsert(List.of(record("gpt-4o", "order_query", 1, 100, 50, "0.0100")));
        mapper.batchUpsert(List.of(record("gpt-4o", "order_query", 2, 200, 100, "0.0200")));
        SummaryRow totals = mapper.summarizeTotals();
        assertNotNull(totals);
        assertEquals(3, totals.getCallCount());
        assertEquals(300, totals.getInputTokens());
        assertEquals(150, totals.getOutputTokens());
    }

    @Test
    void summarizeByModelAndIntent() {
        mapper.batchUpsert(List.of(
                record("gpt-4o", "order_query", 1, 100, 50, "0.0100"),
                record("gpt-4o", "refund_request", 2, 300, 100, "0.0300"),
                record("deepseek-chat", "general", 1, 50, 10, "0.0010")));
        List<SummaryRow> byModel = mapper.summarizeByModel();
        assertEquals(2, byModel.size());
        SummaryRow gpt = byModel.stream()
                .filter(row -> "gpt-4o".equals(row.getModel()))
                .findFirst()
                .orElseThrow();
        assertEquals(3, gpt.getCallCount());
        List<SummaryRow> byIntent = mapper.summarizeByIntent();
        assertEquals(3, byIntent.size());
    }
}
