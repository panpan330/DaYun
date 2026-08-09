package com.panpan.aibusinessservice.entity;

import java.time.Instant;

public class RagEvalRun {
    private String runId;
    private String retriever;
    private Instant startedAt;
    private Instant completedAt;
    private int topK;
    private double hitRate;
    private double recall;
    private double precision;
    private double mrr;
    private int caseCount;
    private String detailsJson;
    private Instant createdAt;

    public String getRunId() { return runId; }
    public void setRunId(String runId) { this.runId = runId; }
    public String getRetriever() { return retriever; }
    public void setRetriever(String retriever) { this.retriever = retriever; }
    public Instant getStartedAt() { return startedAt; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }
    public Instant getCompletedAt() { return completedAt; }
    public void setCompletedAt(Instant completedAt) { this.completedAt = completedAt; }
    public int getTopK() { return topK; }
    public void setTopK(int topK) { this.topK = topK; }
    public double getHitRate() { return hitRate; }
    public void setHitRate(double hitRate) { this.hitRate = hitRate; }
    public double getRecall() { return recall; }
    public void setRecall(double recall) { this.recall = recall; }
    public double getPrecision() { return precision; }
    public void setPrecision(double precision) { this.precision = precision; }
    public double getMrr() { return mrr; }
    public void setMrr(double mrr) { this.mrr = mrr; }
    public int getCaseCount() { return caseCount; }
    public void setCaseCount(int caseCount) { this.caseCount = caseCount; }
    public String getDetailsJson() { return detailsJson; }
    public void setDetailsJson(String detailsJson) { this.detailsJson = detailsJson; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
