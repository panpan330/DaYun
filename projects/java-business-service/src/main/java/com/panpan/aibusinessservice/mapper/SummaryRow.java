package com.panpan.aibusinessservice.mapper;

import java.math.BigDecimal;

public class SummaryRow {
    private String model;
    private String intent;
    private int callCount;
    private long inputTokens;
    private long outputTokens;
    private long totalTokens;
    private BigDecimal estimatedCost;

    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
    public String getIntent() { return intent; }
    public void setIntent(String intent) { this.intent = intent; }
    public int getCallCount() { return callCount; }
    public void setCallCount(int callCount) { this.callCount = callCount; }
    public long getInputTokens() { return inputTokens; }
    public void setInputTokens(long inputTokens) { this.inputTokens = inputTokens; }
    public long getOutputTokens() { return outputTokens; }
    public void setOutputTokens(long outputTokens) { this.outputTokens = outputTokens; }
    public long getTotalTokens() { return totalTokens; }
    public void setTotalTokens(long totalTokens) { this.totalTokens = totalTokens; }
    public BigDecimal getEstimatedCost() { return estimatedCost; }
    public void setEstimatedCost(BigDecimal estimatedCost) { this.estimatedCost = estimatedCost; }
}
