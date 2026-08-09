<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>系统配置摘要</span>
        <el-tag type="warning">不展示密钥原文</el-tag>
      </div>
    </template>

    <el-descriptions :column="2" border>
      <el-descriptions-item label="Java API">VITE_JAVA_API_BASE_URL</el-descriptions-item>
      <el-descriptions-item label="AI API">VITE_AI_API_BASE_URL</el-descriptions-item>
      <el-descriptions-item label="LLM">由 Python AI 服务读取 .env</el-descriptions-item>
      <el-descriptions-item label="Embedding">由 Python AI 服务读取 .env</el-descriptions-item>
      <el-descriptions-item label="Rerank">由 Python AI 服务读取 .env</el-descriptions-item>
      <el-descriptions-item label="Qdrant">由 Python AI 服务连接</el-descriptions-item>
      <el-descriptions-item label="MySQL">由 Java 服务连接</el-descriptions-item>
      <el-descriptions-item label="Redis">由 Java 服务连接</el-descriptions-item>
    </el-descriptions>
  </el-card>
</template>
