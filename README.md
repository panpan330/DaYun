# 答云 DaYun —— 智能客服与工单协作平台

> Smart Customer Service & Ticket Collaboration Platform

一个全栈多 Agent 智能客服系统：**Python (FastAPI + LangGraph)** 承载 AI 能力（多 Agent 意图路由、RAG 知识问答、工单/退款/取消自动化、情绪识别与人工转接、自动化评测），**Java (Spring Boot)** 承载真实业务（订单/工单/用户/知识库元数据），**Vue 3** 提供客户对话与坐席工作台。

## 核心能力

| 模块 | 能力 |
| --- | --- |
| **多 Agent 智能对话** | Supervisor 分派 + Worker 子图（知识/订单/工单/退款/取消），LLM 路由失败自动降级规则 |
| **RAG 知识问答** | 混合检索（关键词+向量）、Rerank、权限过滤、引用溯源、PDF 知识库入库 |
| **工单自动化** | 意图分类 → 字段提取 → 用户确认 → 创建/退款/取消执行，全流程幂等 |
| **情绪识别与转人工** | 愤怒/焦虑/急切自动转人工（多 Agent 与单 Agent 双路径） |
| **坐席协作** | 转人工队列、接管会话、**坐席转移**、**LLM 会话摘要**、人工回复回写客户会话 |
| **运营分析** | Dashboard：满意度、转人工量、情绪分布、每日对话量、LLM 成本、评估快照 |
| **工程化** | 1900+ 测试（TDD）、**自建 pytest e2e 框架**（自动起停三服务跑真实链路）、可观测性（OTel/OpenTelemetry）、幂等/限流/安全边界 |

## 技术栈

- **AI 服务**：Python 3.12 / FastAPI / LangGraph / LangChain / Qdrant(向量) / Redis / httpx
- **业务服务**：Java 21 / Spring Boot 3 / MyBatis / MySQL / H2(测试)
- **前端**：Vue 3 / TypeScript / Element Plus / Vite
- **基础设施**：MySQL / Redis / Qdrant / Milvus（可选）/ OpenTelemetry Collector

## 架构

```
客户 (Vue AiChatView) ──► AI 服务 (Python :8000)
                              ├─ Supervisor 多 Agent 图（意图路由 + Worker 子图）
                              ├─ RAG 检索（Qdrant + 关键词 + Rerank + 权限过滤）
                              ├─ 情绪识别 → 自动转人工
                              └─ 会话存储（Redis + 异步批刷 MySQL）
坐席 (Vue TicketWorkbench) ──► AI 服务（接管/摘要/人工回复） ──► 业务服务 (Java :18004)
业务服务（Java）──► MySQL（工单/订单/转人工/会话/反馈/成本）
```

## 快速开始

### 前置

- Java 21 + Maven、Python 3.12（`uv`）、Node.js 18+
- MySQL、Redis、Qdrant（可用 `docker compose up -d qdrant` 或远程基础设施）

### 1. AI 服务

```bash
cd projects/ai-service
uv sync
cp .env.example .env   # 按需填 LLM_API_KEY / 基础设施地址
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. 业务服务

```bash
cd projects/java-business-service
mvn spring-boot:run    # :18004，自动建表（schema.sql）
```

### 3. 前端

```bash
cd projects/customer-service-console
npm install
npm run dev            # http://localhost:5173
```

### 验证

```bash
# 单测（AI 服务）
cd projects/ai-service && uv run pytest -q

# 端到端（真实服务 + 基础设施在跑）
cd projects/ai-service && uv run pytest tests/e2e -m e2e --run-e2e
```

登录：客户 `U1001`、坐席 `A1001`、主管 `S1001`（开发默认，见 `application.yml` / 前端登录页）。

## 项目结构

```
projects/
  ai-service/                  # AI 服务（FastAPI + LangGraph）
    app/agents/                # 多 Agent 图、意图/情绪分类、worker 子图
    app/rag/                   # RAG：检索/rerank/权限/知识库/评测
    app/services/              # LLM、工具调用、会话存储、Java 客户端
    app/routers/               # HTTP 接口（chat/rag/eval/ops-stats/...）
    tests/                     # 1700+ 单测 + tests/e2e 真实链路
  java-business-service/       # 业务服务（Spring Boot + MySQL）
  customer-service-console/    # 前端（Vue 3）
scripts/run_regression.py      # 一键回归（Python + Java 全量测试）
docs/                          # 接口契约、数据库设计、本地运行指南
```

## 测试

- **单测**：AI 服务 1700+、Java 130+，覆盖 agent 节点/工具/接口/评测/成本等
- **e2e**：`tests/e2e` 自动起停 Java/Python/MCP 三服务，跑真实链路（工单创建、转人工全流程、RAG 问答、会话持久化回源、运营聚合）
- e2e 曾捕获真实生产 bug（Java 侧 `senderType` 校验缺 `human_agent` 导致人工回复批刷失败）

## 文档

- [本地运行与演示指南](docs/local-run-and-demo.md)
- [Java ↔ AI 接口契约](docs/java-ai-api-contract.md)
- [数据库设计](docs/java-business-database-design.md)

## License

MIT
