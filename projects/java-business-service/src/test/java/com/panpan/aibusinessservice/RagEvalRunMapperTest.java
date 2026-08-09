package com.panpan.aibusinessservice;

import com.panpan.aibusinessservice.entity.RagEvalRun;
import com.panpan.aibusinessservice.mapper.RagEvalRunMapper;
import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

@SpringBootTest
class RagEvalRunMapperTest {

    @Autowired
    private RagEvalRunMapper mapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void cleanTables() {
        jdbcTemplate.update("DELETE FROM rag_eval_runs");
    }

    private RagEvalRun sample(String runId, String retriever) {
        Instant now = Instant.now();
        RagEvalRun run = new RagEvalRun();
        run.setRunId(runId);
        run.setRetriever(retriever);
        run.setStartedAt(now);
        run.setCompletedAt(now);
        run.setTopK(5);
        run.setHitRate(0.8);
        run.setRecall(0.75);
        run.setPrecision(0.7);
        run.setMrr(0.6);
        run.setCaseCount(12);
        run.setDetailsJson("[]");
        return run;
    }

    @Test
    void insertAndListRecent() {
        mapper.insert(sample("run-1", "keyword"));
        mapper.insert(sample("run-2", "vector"));

        List<RagEvalRun> runs = mapper.listRecent(10);

        assertEquals(2, runs.size());
    }

    @Test
    void insertIsIdempotentOnDuplicateKey() {
        mapper.insert(sample("run-1", "keyword"));
        mapper.insert(sample("run-1", "keyword"));

        // 幂等语义：重复写入不产生重复行（ON DUPLICATE KEY UPDATE 返回值在 H2 不同模式下不稳定，不依赖）
        assertEquals(1, mapper.listRecent(10).size());
    }

    @Test
    void latestByRetriever() {
        mapper.insert(sample("run-1", "keyword"));
        RagEvalRun run2 = sample("run-2", "keyword");
        run2.setStartedAt(run2.getStartedAt().plusSeconds(5));
        mapper.insert(run2);

        RagEvalRun latest = mapper.latestByRetriever("keyword");

        assertEquals("run-2", latest.getRunId());
    }

    @Test
    void latestByRetrieverReturnsNullWhenNone() {
        assertNull(mapper.latestByRetriever("keyword"));
    }
}
