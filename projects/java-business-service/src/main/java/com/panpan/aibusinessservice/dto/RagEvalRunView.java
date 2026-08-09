package com.panpan.aibusinessservice.dto;

import java.time.Instant;

public record RagEvalRunView(
        String runId,
        String retriever,
        Instant startedAt,
        Instant completedAt,
        int topK,
        double hitRate,
        double recall,
        double precision,
        double mrr,
        int caseCount,
        String detailsJson,
        Instant createdAt
) {}
