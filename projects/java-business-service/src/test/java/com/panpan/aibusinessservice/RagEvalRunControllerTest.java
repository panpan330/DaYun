package com.panpan.aibusinessservice;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.panpan.aibusinessservice.entity.RagEvalRun;
import com.panpan.aibusinessservice.mapper.RagEvalRunMapper;
import java.time.Instant;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class RagEvalRunControllerTest {

    private static final String CALLER = "ai-service";
    private static final String TOKEN = "local-dev-internal-token";

    private org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder withInternalHeaders(
            org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder request
    ) {
        return request
                .header("X-Trace-Id", "trace-rag-eval-test")
                .header("X-Caller", CALLER)
                .header("X-User-Id", "U1001")
                .header("X-Tenant-Id", "default")
                .header("X-Internal-Token", TOKEN);
    }

    @Autowired
    private MockMvc mvc;

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

    private String validBody(String runId) {
        return "{\"run_id\":\"" + runId + "\",\"retriever\":\"keyword\","
                + "\"started_at\":\"2026-08-08T00:00:00Z\",\"completed_at\":\"2026-08-08T00:00:01Z\","
                + "\"top_k\":5,\"hit_rate\":0.8,\"recall\":0.75,\"precision\":0.7,\"mrr\":0.6,"
                + "\"case_count\":12,\"details_json\":\"[]\"}";
    }

    @Test
    void internalPostPersistsRun() throws Exception {
        mvc.perform(withInternalHeaders(post("/internal/rag-eval-runs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(validBody("run-1")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));

        org.junit.jupiter.api.Assertions.assertEquals(1, mapper.listRecent(10).size());
    }

    @Test
    void internalPostRejectsWithoutInternalHeaders() throws Exception {
        mvc.perform(post("/internal/rag-eval-runs")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(validBody("run-noauth")))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void internalPostRejectsInvalidRetriever() throws Exception {
        mvc.perform(withInternalHeaders(post("/internal/rag-eval-runs"))
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"run_id\":\"run-bad\",\"retriever\":\"bogus\","
                                + "\"started_at\":\"2026-08-08T00:00:00Z\",\"completed_at\":\"2026-08-08T00:00:01Z\","
                                + "\"top_k\":5,\"hit_rate\":0.8,\"recall\":0.75,\"precision\":0.7,\"mrr\":0.6,"
                                + "\"case_count\":12,\"details_json\":\"[]\"}"))
                .andExpect(status().isUnprocessableEntity());
    }

    @Test
    void publicGetListsRuns() throws Exception {
        mapper.insert(sample("run-1", "keyword"));

        mvc.perform(get("/api/rag-eval-runs").param("limit", "10"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.data[0].run_id").value("run-1"));
    }

    @Test
    void publicGetLatestByRetriever() throws Exception {
        mapper.insert(sample("run-1", "keyword"));
        mapper.insert(sample("run-2", "keyword"));

        mvc.perform(get("/api/rag-eval-runs/latest").param("retriever", "keyword"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.data.run_id").value("run-2"));
    }
}
