# AI Service

Python AI 服务项目。阶段 1：FastAPI 服务基础已完成；阶段 2：LLM API 基础调用已完成；阶段 3：LangChain + Java 工具调用基础已完成。

当前 `/chat` 已经从 mock 回复改成 OpenAI-compatible 真实模型调用。没有配置本机 `LLM_API_KEY` 时，接口会返回统一配置错误。

当前 `/langchain-chat` 已经支持用 LangChain `ChatOpenAI` / ChatModel 完成一次普通聊天调用，用于对比原生 SDK 调用和 LangChain 模型调用封装。

当前 `/stream-chat` 已经支持 OpenAI-compatible 流式输出，并通过 SSE 逐块返回模型生成内容。

当前 `/extract-ticket` 已经支持 OpenAI-compatible JSON Mode，并用 Pydantic 校验模型返回的结构化工单字段。

当前 `/langchain-extract-ticket` 已经支持用 LangChain `with_structured_output(TicketExtraction, method="json_mode")` 抽取工单字段，用于对比原生 JSON Mode + Pydantic 和 LangChain 结构化输出封装。

当前 `/tools/langchain` 和 `/tools/langchain/query-order` 已经支持查看 LangChain Tool 元数据，并手动调用包装后的 `query_order` 工具。

当前阶段 3 第 1-22 节已完成 Tool Calling 概念、业务系统安全边界、工具参数和 JSON Schema、结构化输出与 Tool Calling 的边界、fake tool 模拟订单查询、工具调用结果 Pydantic 校验、工具调用错误处理、工具调用权限边界、工具调用幂等性、`projects/java-mock-service` 最小业务服务、Python AI 服务调用 Java mock API、让模型决定是否调用工具、工具调用结果再交给模型总结、敏感操作的用户确认机制、确认后创建工单的完整流程、工具调用日志和 `trace_id` 串联、fake Java API / fake tool 的分层测试策略、LangChain 的框架定位和引入时机、LangChain ChatModel 基础、LangChain Tool 基础、LangChain 结构化输出，以及阶段 3 项目整理。后续会进入企业知识库 RAG 基础。

当前阶段 4 企业知识库 RAG 基础已完成，并新增 `app/rag` 内部包，用于承载 RAG 文档、chunk、加载、切分、metadata、embedding、向量库适配、检索和生成等能力。当前已完成 RAG 项目结构、内部 document/chunk 数据模型、文档加载清洗、chunk 切分、metadata 标准化和校验、fake embedding 生成、OpenAI-compatible 真实 embedding 适配器、embedding 独立配置和批量辅助、Qdrant 写入适配、基础 top_k 检索、payload filter 过滤、score_threshold 低相关过滤、把检索结果交给模型生成回答的最小链路、后端根据 retrieved chunks 生成结构化引用来源、无检索结果时的结构化 `no_context` 兜底、RAG embedding/向量库错误映射、可复用的 RAG fake 测试工具、按 `source` 删除旧 chunks 后重新入库的基础文档维护能力、chunk/retrieval 参数调优报告、关键词检索 + 向量检索的混合检索基础、召回结果的学习版 rerank 重排序、检索结果进入模型前的学习版安全检查、缓存/批处理/超时/降级的学习版性能工具、RAG 检索评测样本和 Hit Rate/Recall/Precision/MRR 指标计算、Qdrant/Milvus 选型对比，以及阶段 4 最终收尾复盘。

当前已准备第一批 RAG 练习知识文档，位于 `data/knowledge_base`，并已支持把 Markdown/txt 文件加载和清洗为 `RagDocument`，再按段落和标题切分为 `RagChunk`，经过 metadata 必备字段校验和 Qdrant payload 白名单处理后，用确定性的 fake embedding 生成向量并通过 Qdrant REST API upsert 到本地向量库。现在也支持把用户问题转成 fake query embedding，并通过 Qdrant Query API 取回带 payload filter 和 score_threshold 的 top_k 检索结果，再把关键词检索结果和向量检索结果按 `chunk_id` 去重、分数归一化和加权融合，得到混合检索候选结果；候选结果还能经过规则 reranker 生成 `rerank_score`、`original_rank`、`rerank_rank` 和 `score_breakdown`。检索结果进入模型前可以经过 `RagSecurityPolicy` 安全检查，识别不允许的 `permission_group`、Prompt Injection 风险和敏感信息风险，只把 `safe_chunks` 交给后续生成链路。RAG 性能学习工具现在能构造不暴露原始 query 的检索缓存 key，演示内存 TTL 缓存、batch plan、near_timeout 判断和降级决策。检索结果可以整理成模型上下文生成基于资料的回答，同时返回 `RagCitation` 组成的结构化来源列表；如果没有可用 retrieved chunks，则返回 `status=no_context`、空 citations 和固定 suggestions，不调用模型硬答。查询和入库链路现在会把 embedding 失败、embedding 返回结构异常、向量库调用失败和 collection 配置不匹配映射成统一 RAG 错误码。测试侧新增 `FakeEmbeddingModel`、`FakeVectorStoreReader`、`FakeVectorStoreWriter` 等复用工具，避免单元测试依赖真实 Qdrant、真实 embedding 或真实模型。现在还提供固定评测样本和本地关键词检索评测脚本，用 Hit Rate@K、Recall@K、Precision@K、MRR 和 bad case 报告判断检索质量。

另提供可切换的 `MilvusVectorStore`，能用同一批知识库文档创建 Milvus schema、向量索引和常用 metadata scalar index，把 `EmbeddedChunk` 写成 Milvus entity，并通过 PyMilvus `search` 查回统一的 `RetrievedChunk`。Milvus 适配器已支持 exact match、`in`、整数范围、`should` 和 `must_not` 过滤表达式，并能给已有 collection 补齐缺失的 `INVERTED` scalar index。链路已在基础设施 Docker 里的 Milvus Standalone 上完成入库、检索和 filter/index smoke 验证。当前选型：Qdrant 为 RAG 主线 vector store，Milvus 作为可切换 adapter 与大规模场景对比能力保留。

## 当前能力

- FastAPI 应用创建和启动
- `/health` 健康检查接口
- `/chat` OpenAI-compatible 真实模型聊天接口
- router 路由拆分
- Pydantic 请求模型和响应模型
- `.env` 配置读取
- OpenAI SDK 依赖
- LangChain `langchain-openai` 依赖
- OpenAI-compatible LLM client 初始化
- LangChain `ChatOpenAI` / ChatModel 初始化
- `system` / `user` / `assistant` 消息结构
- prompt 分段构建工具
- `/chat` 真实模型调用
- `/langchain-chat` LangChain ChatModel 普通聊天调用
- `/chat` 可选多轮对话 `history`
- `/stream-chat` 流式聊天接口
- 会话转录持久化：Redis 热缓存 + 异步批刷落 Java（MySQL）永久保存，读取 miss 自动回源重建（Redis 只当热数据）
- 上下文窗口：主 Agent 链路把最近 10 轮（`AGENT_HISTORY_MAX_ROUNDS`，token 预算兜底）喂给意图分类模型，打开旧会话能语义接续
- SSE `message` / `done` / `error` 事件格式
- `/extract-ticket` 工单字段结构化抽取接口
- `/langchain-extract-ticket` LangChain 结构化工单字段抽取接口
- `/tools/query-order` 订单查询工具接口，当前调用 Java mock API
- `/tools/langchain` LangChain Tool 元数据查看接口
- `/tools/langchain/query-order` 手动调用 LangChain `query_order` Tool 的学习接口
- `/tool-decision` 模型工具调用决策接口，当前只返回工具调用意图，不执行工具
- `/tool-chat` 完整工具调用接口：执行 `query_order` 后把 tool message 交回模型生成最终回答
- `/tools/confirmations` 创建待确认的写操作计划，不执行工具
- `/tools/confirmations/{confirmation_id}/confirm` 由同一操作者确认已绑定的计划，仍不执行工具
- `/tickets/plans` 从用户问题提取工单字段并创建 `create_ticket` 确认计划
- `/tickets/confirmations/{confirmation_id}/execute` 执行已确认的创建工单计划
- Pydantic 结构化输出模型和 JSON Schema 生成
- 模型返回 JSON 的 Pydantic 校验
- `QueryOrderArgs` 工具参数模型
- `QueryOrderResult` fake 工具返回模型
- `query_order` 订单查询工具函数
- `JavaOrderClient` Java mock 订单服务 HTTP 客户端
- `map_java_order_to_query_order_payload` Java 订单响应到 AI 工具结果的字段映射函数
- `JAVA_MOCK_SERVICE_BASE_URL` 和 `JAVA_MOCK_SERVICE_TIMEOUT_SECONDS` 跨服务调用配置
- `validate_query_order_result` 工具结果校验函数
- `TOOL_RESULT_VALIDATION_FAILED` 工具结果校验失败错误
- `map_query_order_error` 工具底层异常映射函数
- `TOOL_TIMEOUT`、`TOOL_UPSTREAM_ERROR`、`TOOL_CALL_FAILED` 工具调用错误
- `ToolDefinition` 工具定义模型
- `ToolAccessLevel` 工具风险等级
- `TOOL_REGISTRY` 后端工具注册表
- `authorize_tool_call` 工具权限守卫函数
- `TOOL_NOT_ALLOWED`、`TOOL_CONFIRMATION_REQUIRED` 工具权限错误
- `StructuredTool.from_function()` LangChain Tool 包装
- `QueryOrderArgs` 作为 LangChain Tool `args_schema`
- `LangChainToolInfo` 和 `LangChainToolListResponse` 工具元数据响应模型
- `Idempotency-Key` 工具调用幂等请求头
- `run_idempotent_tool` 工具幂等执行包装函数
- `build_arguments_fingerprint` 工具名和参数指纹生成函数
- `IDEMPOTENCY_KEY_CONFLICT`、`IDEMPOTENCY_KEY_INVALID` 工具幂等错误
- `ToolDecisionType` 工具决策枚举
- `ToolCallCandidate` 模型请求的工具调用候选
- `ToolDecisionResponse` 工具决策响应模型
- `ToolDecisionService` 模型工具选择服务
- `LangChainChatModelService` LangChain ChatModel 调用服务
- `LangChainStructuredOutputService` LangChain 结构化输出服务
- `with_structured_output(TicketExtraction, method="json_mode")` 结构化输出封装
- `ToolCallingChatService` 一次只读工具调用与第二轮模型总结服务
- `SystemMessage`、`HumanMessage`、`AIMessage` 消息对象转换
- `model.invoke()` 一次完整 ChatModel 调用
- `tools=...` 和 `tool_choice="auto"` 模型工具选择参数
- `tool_calls` 模型工具调用请求解析
- `TOOL_ARGUMENTS_INVALID_JSON`、`TOOL_ARGUMENTS_VALIDATION_FAILED`、`TOOL_DECISION_TOO_MANY_CALLS` 工具决策阶段错误
- `TOOL_CALL_ID_MISSING`、`TOOL_SUMMARY_UNEXPECTED_TOOL_CALL` 完整工具调用阶段错误
- `ToolConfirmationService`、`ToolConfirmationStore` 待确认操作的状态管理
- 确认 ID、操作者绑定、参数指纹、过期时间和确认幂等
- `TOOL_CONFIRMATION_NOT_FOUND`、`TOOL_CONFIRMATION_FORBIDDEN`、`TOOL_CONFIRMATION_EXPIRED` 确认阶段错误
- `CreateTicketArgs`、`CreatedTicket` 工单创建命令和 Java 返回结果模型
- `TicketWorkflowService` 工单计划、确认计划消费和执行编排
- `JavaTicketClient` Java mock 工单服务 HTTP 客户端
- 创建工单时使用 `confirmation_id` 作为幂等键
- `TICKET_INTENT_UNSUPPORTED`、`TICKET_ARGUMENTS_VALIDATION_FAILED`、`TICKET_UPSTREAM_REJECTED` 工单流程错误
- `build_trace_headers` 出站 `X-Trace-Id` 构建函数
- Java 订单/工单 HTTP client 会把当前 `trace_id` 传给下游
- `tool_execution_*`、`ticket_execution_*`、`java_order_request_*`、`java_ticket_create_*` 关键节点日志
- 工具调用日志只记录定位字段，不记录完整用户问题、完整工单描述或 API key
- 共享 fake OpenAI-compatible client 测试工具
- 共享 fake 订单查询、工单字段提取和工单创建测试工具
- service/client/router 分层测试策略
- `httpx.MockTransport` 模拟 Java API HTTP 响应
- FastAPI `dependency_overrides` 替换接口依赖
- 模型调用 timeout 统一错误处理
- SDK retry 次数配置和 rate limit 统一错误处理
- OpenAI-compatible SDK 常见错误映射
- 模型调用成功/失败日志和流式调用日志
- 模型响应 `usage` token 用量提取
- fake LLM service/client 测试隔离
- logging 基础日志
- `trace_id` 请求追踪
- 统一异常处理
- CORS 基础配置
- token 粗略估算和输出 token 上限配置
- pytest 自动化测试
- `app/rag` RAG 内部包边界
- `RagDocument` 和 `RagChunk` 内部数据模型
- `data/knowledge_base` 第一批 Markdown/txt RAG 练习文档
- Markdown/txt/PDF 文档加载和基础清洗（PDF：pymupdf 文本提取 + 扫描页 RapidOCR 兜底 + 页眉页脚/页码/断词/OCR 噪声清理；PyMuPDF 为 AGPL 双授权，学习/开源用途）
- 知识库文件上传（md/txt/pdf）：`POST /api/knowledge-base/documents/upload`，前端知识库页"上传文档"选择文件即可落盘入库并同步 Java 元数据
- 段落优先、标题感知的基础 chunk 切分
- metadata 标准化、必备字段校验和 Qdrant payload 白名单
- deterministic fake embedding 生成
- OpenAI-compatible 真实 embedding 适配器、独立配置和 batch helper
- Qdrant collection 校验、point 组装、upsert 写入和按 payload filter 删除 points
- Milvus schema 创建、向量索引创建、常用 metadata scalar index 创建、entity upsert、flush-on-wait 和 PyMilvus search 检索
- Milvus metadata filter：exact match、`in`、整数范围、`should`、`must_not` 转换为 Milvus boolean expression
- 最小 RAG 入库流程：load -> split -> embed -> store
- 文档删除和重新入库：按 `source` 删除旧 chunks，再 upsert 新 chunks
- 基础 top_k 检索：query -> fake query embedding -> Qdrant Query API -> RetrievedChunk
- 基础 payload filter：按 `permission_group`、`business_domain`、`doc_type`、`source` 限定检索范围
- 基础 score_threshold：过滤低相关检索结果，给后续“无资料不回答”打基础
- RAG 调优报告：观察 `chunk_size`、`chunk_overlap`、`top_k`、`score_threshold` 对切分和检索结果的影响
- RAG 混合检索：关键词检索 + 向量检索合并、去重、分数归一化和加权融合
- RAG rerank 重排序：召回候选 -> 规则打分 -> 保留原始排名、重排排名和分数拆解
- RAG 安全检查：permission_group 复查、Prompt Injection 检测、敏感信息识别、safe chunks 过滤
- RAG 性能基础：检索缓存 key、内存 TTL cache、batch plan、near_timeout、降级决策
- RAG 检索评测：固定评测样本、Hit Rate@K、Recall@K、Precision@K、MRR@K 和 bad case 报告
- RAG 权限过滤：对话链路按用户角色映射权限组检索（无权限明确提示 `permission_denied`，不泄露文档存在性）
- RAG 高级检索默认启用：知识路由（collection 选择）+ 混合检索 + 查询改写（`rag_enable_*` 开关可关）
- RAG 检索评测落库：`scripts/rag_retrieval_eval.py --save-run` 双模式结果同步 Java（`rag_eval_runs` 表），评测页历史卡片 + 趋势图
- RAG 阶段复盘：入库链路、问答链路、模块职责、Qdrant/Milvus 选型、评测和生产级差距
- RAG 主线项目复盘：入库流水线、问答流水线、模块职责、学习版/生产级差距和验收清单
- 基础 RAG 生成：RetrievedChunk -> RAG context -> OpenAI-compatible model -> grounded answer
- 基础 RAG 引用来源：RetrievedChunk -> backend-generated RagCitation -> RagAnswer(answer, citations)
- 基础 RAG 无资料兜底：chunks=[] -> RagAnswer(status=no_context, citations=[], suggestions=[...])
- 基础 RAG 错误处理：embedding/vector store failures -> AppException RAG_* 错误码
- RAG fake 测试工具：FakeEmbeddingModel、FakeVectorStoreReader、FakeVectorStoreWriter、make_retrieved_chunk
- 情绪识别与情绪驱动转人工：7 种情绪（angry/anxious/dissatisfied/urgent/apologetic/neutral/satisfied），rule_based/fake_llm/real_llm 三模式；angry/anxious/urgent 自动安抚并转人工（跳过写工具）；emotion 评测套件纳入 agent eval
- 多轮对话评测：multiturn 评测套件（逐轮意图/路由断言 + 跨轮引用严格模式），`data/agent_eval/multiturn_cases.json` 独立多轮用例库
- 成本可视化：LLM 调用成本按模型 × 意图聚合（进程内 CostRecorder → 周期同步 Java `ai_cost_records` 表），Dashboard 成本概览卡片 + `/api/ai/cost/overview`；token 用量始终记录，费用按 `LLM_INPUT_COST/LLM_OUTPUT_COST` 估算（未配置时为 0）
- 工具动态按需注入：规则+LLM 混合感知（订单号/业务动词/历史增强 → LLM 意图兜底），写工具需"动词+订单号"双信号，兜底只读 query_order；`TOOL_SELECTION_ENABLED` 开关（默认开，false 回退全量）

## 项目结构

```text
app/
  core/
    config.py              配置读取
    cors.py                CORS 配置
    exception_handlers.py  统一异常处理器
    exceptions.py          项目业务异常
    logging.py             日志配置
    token_usage.py         token 粗略估算和预算辅助
    trace.py               trace_id 上下文
  middleware/
    tracing.py             请求追踪 middleware
  routers/
    chat.py                /chat 路由
    health.py              /health 路由
    tickets.py             工单计划和已确认计划执行路由
    tools.py               工具调用学习接口
  rag/
    README.md              RAG 内部包职责说明和后续模块规划
    documents.py           RAG 文档和 chunk 内部数据模型
    loaders.py             Markdown/txt 文档加载和基础清洗
    splitters.py           RagDocument 到 RagChunk 的基础切分
    metadata.py            metadata 标准化、校验和 Qdrant payload 构造
    embeddings.py          fake embedding 模型、OpenAI-compatible embedding 适配器、batch helper 和 EmbeddedChunk
    filters.py             Qdrant payload filter 构造
    vector_store.py        Qdrant point 组装、collection 校验、upsert 写入、按 filter 删除 points 和 query 检索
    milvus_store.py        Milvus entity 组装、schema/index 创建、scalar filter、upsert 写入和 search 检索
    ingestion.py           load -> split -> embed -> store 入库编排、文档删除和目录刷新
    retriever.py           query embedding、payload filter、score_threshold 和 top_k 检索编排
    generator.py           把检索结果整理成上下文，调用模型生成回答，构造结构化引用来源，并处理 no_context 兜底
    tuning.py              chunk 切分参数和检索参数调优报告
    hybrid.py              简单关键词检索和关键词 + 向量结果融合
    rerank.py              召回候选的学习版规则重排序
    security.py            检索结果进入模型前的学习版安全检查
    performance.py         缓存、批处理、超时和降级的学习版性能工具
    evaluation.py          检索评测样本、指标计算和 bad case 报告
    errors.py              RAG embedding 和 vector store 错误映射
  schemas/
    chat.py                聊天请求/响应模型
    error.py               统一错误响应模型
    structured.py          结构化输出请求/响应和工单字段模型
    ticket.py              创建工单命令、响应和工作流请求模型
    tool.py                工具参数和工具结果模型
    tool_decision.py       模型工具调用决策响应模型
    tool_confirmation.py   工具确认请求、状态和响应模型
  services/
    langchain_chat_model_service.py LangChain ChatModel 调用服务
    langchain_structured_output_service.py LangChain 结构化输出服务
    llm_client.py          OpenAI-compatible SDK client 初始化
    llm_service.py         LLM 聊天调用服务
    message_builder.py     聊天 messages 构建工具
    prompt_builder.py      prompt 分段构建工具
    structured_output_service.py 结构化输出调用服务
    java_order_client.py   Java mock 订单服务 HTTP 客户端
    java_ticket_client.py  Java mock 工单服务 HTTP 客户端
    ticket_workflow_service.py 工单字段提取、确认计划消费和创建工单编排
    tool_decision_service.py 模型工具选择和 tool_calls 解析服务
    tool_calling_chat_service.py 执行工具并将结果回传模型总结
    tool_confirmation_service.py 创建和确认待执行工具计划
  tools/
    fake_order_tool.py     订单查询工具，当前调用 Java mock API
    idempotency.py         工具调用幂等性辅助函数
    langchain_tools.py     LangChain Tool 适配和元数据辅助
    tool_registry.py       工具注册表和权限守卫
    tool_confirmation.py   内存确认计划存储与过期检查
  main.py                  FastAPI 应用入口
data/
  knowledge_base/
    README.md              RAG 示例知识库说明
    account-security-faq.md 账号安全 FAQ 示例文档
    logistics-tracking-faq.txt 物流查询 FAQ 示例文档
    order-shipping-policy.md 订单发货规则示例文档
    refund-return-policy.md 退款退货规则示例文档
  rag_eval/
    README.md              RAG 检索评测样本说明
    retrieval_cases.json   固定检索评测样本
scripts/
  llm_compatible_smoke_test.py 手动检查或调用兼容模型
  rag_ingest_smoke.py     手动执行 fake embedding 到 Qdrant 的入库烟测
  rag_retrieve_smoke.py   手动执行 fake query embedding 的过滤检索烟测
  rag_milvus_smoke.py     手动执行 fake embedding 到 Milvus 的入库和检索烟测
  rag_milvus_filter_smoke.py 手动执行 Milvus metadata filter 和 scalar index 烟测
  rag_chunk_tuning_preview.py 不连接 Qdrant 的 chunk 切分参数预览
  rag_keyword_search_preview.py 不连接 Qdrant 的关键词检索预览
  rag_rerank_preview.py   不连接 Qdrant 的 rerank 重排序预览
  rag_security_preview.py 不连接 Qdrant 的 RAG 安全检查预览
  rag_performance_preview.py 不连接 Qdrant 的 RAG 性能工具预览
  rag_retrieval_eval.py   手动执行本地关键词检索评测烟测
tests/
  conftest.py              pytest 共享夹具
  fakes.py                 OpenAI-compatible fake client 测试工具
  rag_fakes.py             RAG fake embedding、fake vector store 和 RetrievedChunk 构造工具
  tool_fakes.py            工具调用 fake 对象和测试数据构造工具
  test_chat_api.py         /chat 接口测试
  test_chat_schema.py      聊天模型测试
  test_config.py           配置测试
  test_cors.py             CORS 测试
  test_exception_handlers.py 统一异常处理测试
  test_fake_order_tool.py  订单查询工具映射和校验测试
  test_java_order_client.py Java mock HTTP 客户端测试
  test_java_ticket_client.py Java mock 工单 HTTP 客户端测试
  test_fake_llm_client.py  fake LLM client 工具测试
  test_health.py           /health 测试
  test_langchain_chat_model_service.py LangChain ChatModel 服务测试
  test_langchain_structured_output_service.py LangChain 结构化输出服务测试
  test_langchain_tools.py  LangChain Tool 适配测试
  test_llm_client.py       LLM client 初始化测试
  test_llm_service.py      LLM 聊天服务测试
  test_logging.py          日志测试
  test_message_builder.py  聊天 messages 构建测试
  test_prompt_builder.py   prompt 分段构建测试
  test_rag_documents.py    RAG 文档和 chunk 内部模型测试
  test_knowledge_base_samples.py 示例知识库文件存在性测试
  test_rag_loaders.py      RAG Markdown/txt 文档加载测试
  test_rag_splitters.py    RAG chunk 切分测试
  test_rag_metadata.py     RAG metadata 标准化、校验和 payload 白名单测试
  test_rag_embeddings.py   RAG embedding 生成测试
  test_rag_filters.py      RAG payload filter 构造测试
  test_rag_vector_store.py Qdrant point 组装、写入、删除和 query 适配测试
  test_rag_milvus_store.py Milvus entity、schema/index、scalar filter、upsert 和 search 适配测试
  test_rag_ingestion.py    RAG 入库、文档删除和重新入库编排测试
  test_rag_retriever.py    RAG query embedding、payload filter、score_threshold 和 top_k 检索编排测试
  test_rag_generator.py    RAG 上下文构造、模型生成、结构化引用来源和 no_context 兜底测试
  test_rag_tuning.py       RAG chunk 参数和检索参数调优报告测试
  test_rag_hybrid.py       RAG 关键词检索和混合检索融合测试
  test_rag_rerank.py       RAG 召回结果重排序测试
  test_rag_security.py     RAG 权限、Prompt Injection 和敏感信息安全检查测试
  test_rag_performance.py  RAG 缓存、批处理、超时和降级决策测试
  test_rag_evaluation.py   RAG 检索评测样本、指标计算和 bad case 报告测试
  test_rag_errors.py       RAG embedding 和 vector store 错误映射测试
  test_rag_fakes.py        RAG fake 测试工具自身行为测试
  test_structured_output_service.py 结构化输出服务测试
  test_structured_schema.py 结构化输出模型测试
  test_tool_idempotency.py 工具调用幂等性测试
  test_tool_registry.py    工具注册表和权限守卫测试
  test_tool_decision_schema.py 工具决策响应模型测试
  test_tool_decision_service.py 模型工具选择服务测试
  test_tool_fakes.py       共享工具 fake 测试
  test_tool_calling_chat_service.py 完整工具调用与第二轮总结服务测试
  test_tool_confirmation_schema.py 确认请求模型测试
  test_tool_confirmation_service.py 确认计划、操作者绑定和过期测试
  test_ticket_workflow_service.py 工单计划、确认消费和执行编排测试
  test_tickets_api.py      工单计划和执行接口测试
  test_tool_schema.py      工具参数和工具结果模型测试
  test_tools_api.py        /tools/query-order 接口测试
  test_token_usage.py      token 粗略估算测试
  test_trace.py            trace_id 测试
```

## 运行

首次进入项目时，先同步依赖：

```powershell
uv sync
```

如果需要本地配置，先复制示例配置文件：

```powershell
Copy-Item .env.example .env
```

再启动服务：

```powershell
uv run uvicorn app.main:app --reload
```

启动后访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 测试

```powershell
uv run pytest -q
```

当前测试使用 FastAPI 的 `TestClient`，覆盖 `/health`、`/chat`、`/langchain-chat`、`/stream-chat`、`/extract-ticket`、`/langchain-extract-ticket`、`/tool-decision`、`/tool-chat`、`/tools/query-order`、`/tools/langchain`、`/tools/langchain/query-order`、`/tools/confirmations`、`/tickets/plans`、`/tickets/confirmations/{confirmation_id}/execute`、`ChatRequest`、`ChatResponse`、`ChatMessage`、`TicketExtraction`、`CreateTicketArgs`、`CreatedTicket`、`QueryOrderArgs`、`QueryOrderResult`、`ToolDefinition`、`ToolAccessLevel`、`ToolDecisionType`、`ToolCallCandidate`、`ToolDecisionResponse`、`LangChainToolInfo`、`LangChainToolListResponse`、确认请求模型、确认计划状态、工单计划和工单执行响应、多轮 `history`、配置读取、日志、`trace_id`、出站 `X-Trace-Id`、统一异常处理、CORS、token 粗略估算、LLM client 初始化、LLM service、LangChain ChatModel service、LangChain structured output service、LangChain Tool 适配、结构化输出 service、工具决策 service、完整工具调用 service、确认计划 service、工单工作流 service、JavaOrderClient、JavaTicketClient、Java mock API 字段映射、Java mock 工单创建、工具结果 Pydantic 校验、工具参数 Pydantic 校验、assistant tool-call message、tool message、`tool_call_id` 关联、LangChain `SystemMessage` / `HumanMessage` / `AIMessage` 转换、`model.invoke()` 调用封装、`with_structured_output(TicketExtraction, method="json_mode")` 结构化输出封装、`StructuredTool.from_function()` 包装、LangChain Tool `args_schema`、操作者/参数绑定、参数指纹、确认过期和确认幂等、确认计划消费、创建工单幂等、工具调用关键节点日志、工具调用 timeout/上游错误映射、工具注册表和权限守卫、模型可见工具筛选、工具调用幂等性、fake OpenAI-compatible client、fake LangChain ChatModel、fake LangChain structured output model、fake LangChain Tool、fake `tool_calls`、共享 fake tool、fake Java API、`httpx.MockTransport`、`dependency_overrides`、OpenAI-compatible SDK 错误映射、模型调用日志、流式调用日志、结构化输出日志、模型响应 token usage 提取、messages 构建和 prompt 构建。

也可以运行 Python 编译检查：

```powershell
uv run python -m compileall -q -x ".venv|__pycache__" .
```

### 端到端测试（真实服务）

前提：本地 MySQL、VM Redis/Qdrant 在跑（Java/Python 依赖的基础设施）。

```powershell
uv run pytest tests/e2e -m e2e --run-e2e
```

- 自动起停 Java（18004）/ Python（8000）/ MCP（9100）三服务；端口已占用时复用
- 覆盖链路：工单创建（internal→MySQL→列表）、转人工全链（触发→claim→transfer→human-reply→summary→close）、RAG 问答、会话持久化（批刷→MySQL→删 Redis→回源）、运营聚合
- 不带 `--run-e2e` 时全部跳过，不影响单测
- 调试选项：`--keep-services` 保留服务进程；服务日志在 `.e2e_logs/`

## 配置

本项目使用 `app/core/config.py` 集中读取配置。

`.env.example` 是可以提交到 GitHub 的示例配置，真实 `.env` 只放在本机，不提交。

| 配置项 | 说明 |
| --- | --- |
| `APP_NAME` | FastAPI 应用名称 |
| `APP_DESCRIPTION` | FastAPI 应用描述 |
| `APP_VERSION` | FastAPI 应用版本 |
| `MODEL_NAME` | 当前使用的模型名称，现阶段先是 mock 名称 |
| `LLM_PROVIDER` | LLM 服务商标识，例如 `aliyun-compatible` |
| `LLM_MODEL` | 真实模型名，例如 `qwen3.7-plus` |
| `LLM_BASE_URL` | OpenAI-compatible 接口地址，真实值只放本机 `.env` |
| `LLM_API_KEY` | LLM API key，真实值只放本机 `.env` 或系统环境变量 |
| `REQUEST_TIMEOUT_SECONDS` | 后续调用模型或外部接口时使用的超时时间 |
| `LLM_MAX_RETRIES` | OpenAI-compatible SDK 自动重试次数，默认 `2`，当前允许 `0-5` |
| `MAX_OUTPUT_TOKENS` | 后续限制模型最多生成多少输出 token |
| `JAVA_MOCK_SERVICE_BASE_URL` | Java mock 订单服务基础地址，默认 `http://127.0.0.1:8001` |
| `JAVA_MOCK_SERVICE_TIMEOUT_SECONDS` | 调用 Java mock 订单服务的超时时间，默认 `5` 秒 |
| `TOOL_CONFIRMATION_TTL_SECONDS` | 待确认计划有效秒数，默认 `300`，当前允许 `30-3600` |
| `AGENT_CHECKPOINT_TTL_MINUTES` | Agent 线程 checkpoint/会话转录在 Redis 的保留分钟数，默认 `43200`（30 天），允许 `1-43200` |
| `CONVERSATION_SYNC_ENABLED` | 是否开启会话转录异步批刷到 Java（MySQL）持久化，默认 `true` |
| `CONVERSATION_SYNC_INTERVAL_SECONDS` | 批刷轮询间隔秒数，默认 `30` |
| `AGENT_HISTORY_MAX_ROUNDS` | 喂给模型的最近对话轮数上限，默认 `10`，允许 `1-50` |
| `AGENT_HISTORY_MAX_TOKENS` | 历史窗口 token 预算上限，默认 `3000`，允许 `200-20000` |
| `RAG_ENABLE_ROUTING` | 高级检索：知识路由（collection 选择），默认 `true` |
| `RAG_ENABLE_HYBRID` | 高级检索：关键词 + 向量混合检索，默认 `true` |
| `RAG_ENABLE_REWRITE` | 高级检索：查询改写，默认 `true` |
| `RAG_ENABLE_MULTI_QUERY` | 高级检索：多查询扩展（放大 embedding 成本），默认 `false` |
| `RAG_ENABLE_CONTEXT_COMPRESSION` | 高级检索：上下文压缩（影响引用完整性），默认 `false` |
| `RAG_ENABLE_CITATION_VERIFY` | 高级检索：引用来源校验（log-only），默认 `false` |
| `LOG_LEVEL` | 日志级别 |
| `CORS_ALLOWED_ORIGINS` | 允许跨源访问后端的前端来源，多个值用逗号分隔 |
| `OPENAI_API_KEY` | 旧版兼容字段，后续优先使用 `LLM_API_KEY` |

`LLM_API_KEY` 和 `OPENAI_API_KEY` 都属于敏感信息。真实值只应该放在本机 `.env` 或系统环境变量里，不要写进代码、README、测试用例、截图或聊天记录里。

项目里可以通过 `settings.has_llm_api_key` 判断是否已经配置了非空 key。`LLM_API_KEY=""` 或全是空格时，都视为未配置。

`app/core/token_usage.py` 提供的是本地粗略估算工具，用来学习和做预算保护，不等于真实计费结果。真实 token 数以后要以模型 API 响应里的 `usage` 为准。

## OpenAI-compatible SDK 检查

当前项目使用官方 `openai` Python SDK 初始化 OpenAI-compatible client。

配置入口：

```text
app/services/llm_client.py
```

手动检查配置：

```powershell
uv run python scripts/llm_compatible_smoke_test.py
```

这条命令默认不会调用模型，只检查本机 `.env` 是否已经配置 `LLM_API_KEY`。

脚本真实调用时会先用 `app/services/prompt_builder.py` 把用户输入整理成包含任务、要求、输出格式和失败策略的清晰 prompt。

确认要真实调用模型时，再显式加：

```powershell
uv run python scripts/llm_compatible_smoke_test.py --call
```

注意：`--call` 会请求真实模型，可能产生费用。真实 key 不要发给任何人，只放本机 `.env`。

## 真实 `/chat`

`/chat` 当前通过 `app/services/llm_service.py` 调用 OpenAI-compatible 模型，并支持可选 `history` 做多轮对话。

调用链路：

```text
POST /chat
-> app/routers/chat.py
-> LLMChatService.generate_reply()
-> prompt_builder.py
-> message_builder.py
-> llm_client.py
-> client.chat.completions.create(...)
```

请求示例：

```json
{
  "message": "请解释 FastAPI 是什么"
}
```

多轮请求示例：

```json
{
  "message": "那 FastAPI 呢？",
  "history": [
    {"role": "user", "content": "什么是 API？"},
    {"role": "assistant", "content": "API 是程序之间约定好的调用方式。"}
  ]
}
```

`history` 只允许 `user` 和 `assistant`，不允许客户端传 `system`。当前最多允许 20 条历史消息。

成功响应示例：

```json
{
  "reply": "模型生成的回答"
}
```

如果没有配置本机 `LLM_API_KEY`，会返回：

```json
{
  "code": "LLM_API_KEY_MISSING",
  "message": "LLM API key 未配置，请先在本机 .env 中配置 LLM_API_KEY。",
  "trace_id": "..."
}
```

如果模型调用超过 `REQUEST_TIMEOUT_SECONDS`，会返回：

```json
{
  "code": "LLM_TIMEOUT",
  "message": "模型调用超时，请稍后重试。",
  "trace_id": "..."
}
```

如果模型服务返回限流，或请求过于频繁，会返回：

```json
{
  "code": "LLM_RATE_LIMITED",
  "message": "模型服务请求过于频繁，请稍后重试。",
  "trace_id": "..."
}
```

自动化测试不会真实调用模型。测试通过 FastAPI `dependency_overrides` 注入 fake service，通过 fake client 测试 `LLMChatService`。

当前模型调用错误会先映射成项目统一错误码，再由统一异常处理器返回 JSON：

| 错误码 | HTTP 状态码 | 含义 |
| --- | --- | --- |
| `LLM_API_KEY_MISSING` | 500 | 本机没有配置模型 API key |
| `LLM_TIMEOUT` | 504 | 模型调用超时 |
| `LLM_RATE_LIMITED` | 429 | 模型服务限流或请求过于频繁 |
| `LLM_AUTHENTICATION_FAILED` | 502 | 模型服务认证失败 |
| `LLM_PERMISSION_DENIED` | 502 | 模型服务拒绝访问 |
| `LLM_RESOURCE_NOT_FOUND` | 502 | 模型、接口或资源不存在 |
| `LLM_BAD_REQUEST` | 502 | 发给模型服务的请求参数错误 |
| `LLM_PROVIDER_ERROR` | 502 | 模型服务内部错误 |
| `LLM_CONNECTION_ERROR` | 502 | 无法连接模型服务 |
| `LLM_PROVIDER_STATUS_ERROR` | 502 | 模型服务返回其他异常状态 |
| `LLM_BAD_RESPONSE` | 502 | 模型返回格式异常 |
| `LLM_EMPTY_RESPONSE` | 502 | 模型返回空内容 |
| `LLM_CALL_FAILED` | 502 | 其他模型调用失败 |
| `STRUCTURED_OUTPUT_EMPTY` | 502 | 模型没有返回可解析的结构化内容 |
| `STRUCTURED_OUTPUT_VALIDATION_FAILED` | 502 | 模型返回的结构化内容不符合 Pydantic 模型 |

## 流式 `/stream-chat`

`/stream-chat` 当前通过 `app/services/llm_service.py` 调用 OpenAI-compatible 模型，并开启：

```python
stream=True
stream_options={"include_usage": True}
```

调用链路：

```text
POST /stream-chat
-> app/routers/chat.py
-> LLMChatService.stream_reply()
-> prompt_builder.py
-> message_builder.py
-> llm_client.py
-> client.chat.completions.create(..., stream=True)
-> StreamingResponse
```

请求体和 `/chat` 相同：

```json
{
  "message": "请解释 FastAPI 是什么"
}
```

响应类型是：

```text
text/event-stream
```

成功时会逐块返回 SSE：

```text
event: message
data: {"content":"FastAPI"}

event: message
data: {"content":" 是 Python Web 框架。"}

event: done
data: {"trace_id":"..."}
```

流开始前发生的错误，例如缺少 `LLM_API_KEY`，仍然返回统一 JSON 错误。

流开始后发生的错误，会返回 SSE `error` 事件：

```text
event: error
data: {"code":"LLM_CALL_FAILED","message":"模型调用失败，请稍后重试。","trace_id":"..."}
```

## 结构化 `/extract-ticket`

`/extract-ticket` 当前通过 `app/services/structured_output_service.py` 调用 OpenAI-compatible 模型，并开启 JSON Mode：

```python
response_format={"type": "json_object"}
```

调用链路：

```text
POST /extract-ticket
-> app/routers/chat.py
-> StructuredOutputService.extract_ticket()
-> build_ticket_extraction_messages()
-> llm_client.py
-> client.chat.completions.create(..., response_format={"type":"json_object"})
-> TicketExtraction.model_validate_json()
```

请求示例：

```json
{
  "message": "订单 A1001 一直没有发货，我要投诉。"
}
```

成功响应示例：

```json
{
  "extraction": {
    "intent": "complaint",
    "order_id": "A1001",
    "summary": "用户投诉订单未发货",
    "urgency": "high",
    "need_human_review": true
  }
}
```

当前支持的 `intent`：

```text
refund
order_query
logistics
complaint
unknown
```

当前支持的 `urgency`：

```text
low
normal
high
```

如果模型返回的 JSON 不符合 `TicketExtraction`，会返回：

```json
{
  "code": "STRUCTURED_OUTPUT_VALIDATION_FAILED",
  "message": "模型结构化输出校验失败，请稍后重试。",
  "trace_id": "...",
  "details": []
}
```

自动化测试不会真实调用模型。接口测试通过 `dependency_overrides` 注入 fake service，服务测试通过 fake client 验证 JSON Mode 参数、Pydantic 解析和错误处理。

## LangChain 结构化 `/langchain-extract-ticket`

`/langchain-extract-ticket` 当前通过 `app/services/langchain_structured_output_service.py` 调用 LangChain ChatModel，并使用 `with_structured_output()` 让模型输出尽量贴合 `TicketExtraction`：

```python
model.with_structured_output(TicketExtraction, method="json_mode")
```

调用链路：

```text
POST /langchain-extract-ticket
-> app/routers/chat.py
-> LangChainStructuredOutputService.extract_ticket()
-> build_langchain_ticket_extraction_messages()
-> create_langchain_chat_model()
-> model.with_structured_output(TicketExtraction, method="json_mode")
-> structured_model.invoke(messages)
-> validate_langchain_ticket_extraction()
```

这个接口和 `/extract-ticket` 解决的是同一类问题：把用户自然语言抽取成结构化工单字段。区别是 `/extract-ticket` 直接使用 OpenAI-compatible SDK、JSON Mode 和手动 Pydantic 解析；`/langchain-extract-ticket` 把结构化输出能力交给 LangChain ChatModel 封装，再由项目自己的校验函数把结果收口成 `TicketExtraction`。

学习边界：

- 当前只学习 LangChain 结构化输出，不替换原来的 `/extract-ticket`。
- 当前不引入 Agent，也不让模型自动执行工具。
- LangChain 可以减少样板代码，但不能替代项目自己的配置读取、异常映射、日志、安全边界和 Pydantic 兜底校验。

自动化测试不会真实调用模型。服务测试通过 fake LangChain model 验证 `with_structured_output(TicketExtraction, method="json_mode")`、`invoke(messages)`、结构化结果校验和错误映射；接口测试通过 `dependency_overrides` 注入 fake service。

## 订单查询工具 `/tools/query-order`

`/tools/query-order` 当前通过 `app/tools/fake_order_tool.py` 执行订单查询工具。文件名还保留 `fake_order_tool.py`，但内部已经不再查询本地内存 fake 数据，而是通过 `app/services/java_order_client.py` 调用 `java-mock-service`：

调用链路：

```text
POST /tools/query-order
-> app/routers/tools.py
-> QueryOrderArgs 校验 order_id
-> authorize_tool_call("query_order")
-> run_idempotent_tool(..., Idempotency-Key)
-> fake_order_tool.query_order()
-> JavaOrderClient.get_order(order_id)
-> GET /orders/{order_id}
-> map_java_order_to_query_order_payload()
-> validate_query_order_result()
-> QueryOrderResult
```

当前默认调用地址来自配置：

```text
JAVA_MOCK_SERVICE_BASE_URL=http://127.0.0.1:8001
JAVA_MOCK_SERVICE_TIMEOUT_SECONDS=5
```

请求示例：

```json
{
  "order_id": "A1001"
}
```

成功响应示例：

```json
{
  "result": {
    "order_id": "A1001",
    "order_status": "waiting_shipment",
    "payment_status": "paid",
    "logistics_message": "商家已接单，等待仓库发货。",
    "latest_event": "仓库正在准备出库。",
    "can_create_ticket": true,
    "source": "java_mock_service"
  }
}
```

响应里不会暴露 Java mock 返回的 `customer_id`。这是工具层字段映射的一部分：只把当前 AI 工具需要的、安全的、稳定的字段返回给模型和调用方。

如果订单号格式合法但 Java mock 服务里不存在，会返回：

```json
{
  "code": "ORDER_NOT_FOUND",
  "message": "订单不存在，请确认订单号是否正确。",
  "trace_id": "..."
}
```

当前 Java mock 服务还提供一个教学用订单号，用来模拟上游服务失败：

| 订单号 | 模拟场景 | 错误码 | HTTP 状态码 |
| --- | --- | --- | --- |
| `A500` | 上游订单服务内部错误 | `TOOL_UPSTREAM_ERROR` | 502 |

如果 `java-mock-service` 没有启动、连接失败或调用超时，`ai-service` 会把底层 HTTP 客户端异常映射成项目统一错误。超时响应示例：

```json
{
  "code": "TOOL_TIMEOUT",
  "message": "订单查询工具调用超时，请稍后重试。",
  "trace_id": "..."
}
```

`A500` 响应示例：

```json
{
  "code": "TOOL_UPSTREAM_ERROR",
  "message": "订单查询服务暂时不可用，请稍后重试。",
  "trace_id": "..."
}
```

当前工具注册表：

| 工具名 | 权限等级 | 是否启用 | 是否需要确认 |
| --- | --- | --- | --- |
| `query_order` | `read` | 是 | 否 |
| `create_ticket` | `write` | 是 | 是 |
| `refund_order` | `sensitive` | 是 | 是 |
| `cancel_order` | `sensitive` | 是 | 是 |

如果模型请求未知工具或禁用工具，后端会返回：

```json
{
  "code": "TOOL_NOT_ALLOWED",
  "message": "工具不在允许列表中，后端已拒绝执行。",
  "trace_id": "..."
}
```

如果工具需要用户确认但当前没有确认，后端会返回：

```json
{
  "code": "TOOL_CONFIRMATION_REQUIRED",
  "message": "该工具需要用户确认后才能执行。",
  "trace_id": "..."
}
```

## 工具调用幂等性

`/tools/query-order` 当前支持可选请求头：

```http
Idempotency-Key: query-order-api-key-001
```

如果不传 `Idempotency-Key`，接口按普通请求执行。

如果传入合法 `Idempotency-Key`：

```text
第一次请求：执行工具并保存工具名、参数指纹和结果。
重复请求且参数相同：返回第一次结果，不再执行工具。
重复请求但参数不同：返回 IDEMPOTENCY_KEY_CONFLICT。
```

当前实现入口：

```text
app/tools/idempotency.py
```

当前是学习用内存版实现，服务重启后记录会丢失。生产环境应升级为数据库或 Redis，并配合唯一索引、TTL、事务和审计日志。

同一个幂等键配不同参数时，后端会返回：

```json
{
  "code": "IDEMPOTENCY_KEY_CONFLICT",
  "message": "同一个幂等键不能用于不同的工具调用参数。",
  "trace_id": "..."
}
```

幂等键格式不合法时，后端会返回：

```json
{
  "code": "IDEMPOTENCY_KEY_INVALID",
  "message": "幂等键格式不正确，请使用 8 到 128 位的字母、数字、点、下划线、冒号或短横线。",
  "trace_id": "..."
}
```

如果 Java mock API 返回的数据不符合 `QueryOrderResult`，会返回：

```json
{
  "code": "TOOL_RESULT_VALIDATION_FAILED",
  "message": "工具返回结果校验失败，请稍后重试。",
  "trace_id": "...",
  "details": []
}
```

当前订单查询工具已经把内部 fake 数据替换成 Java mock API，并完成了模型决策、工具执行、tool message 回传和第二轮模型总结。

## 用户确认计划 `/tools/confirmations`

阶段 3 第 14 节新增的是“确认计划”，不是“确认后立刻执行”。它用于把未来写操作的关键内容固定下来，等待用户明确确认。

调用链路：

```text
POST /tools/confirmations
-> ToolConfirmationRequest(actor_id, tool_name, arguments)
-> require_enabled_tool_definition()
-> 检查工具确实 requires_confirmation=True
-> ToolConfirmationStore 创建 pending 计划
-> 保存操作者、工具名、深拷贝参数、SHA-256 参数指纹、创建/过期时间
-> ToolConfirmationResponse

POST /tools/confirmations/{confirmation_id}/confirm
-> ConfirmToolConfirmationRequest(actor_id)
-> 按确认 ID 读取后端保存的计划
-> 检查操作者相同、计划未过期
-> 状态 pending -> confirmed
-> ToolConfirmationResponse
```

创建确认计划示例：

```json
{
  "actor_id": "demo_user_001",
  "tool_name": "create_ticket",
  "arguments": {
    "title": "订单 A1001 未发货",
    "description": "用户反馈订单迟迟未发货。",
    "order_id": "A1001"
  }
}
```

响应会返回后端生成的 `confirmation_id`、固定参数、参数指纹与过期时间。确认时客户端只能发送操作者：

```json
{
  "actor_id": "demo_user_001"
}
```

确认接口不接受新的 `tool_name` 或 `arguments`。因此用户确认订单 A1001 的创建工单计划后，客户端不能在确认请求里偷偷替换为退款、其他订单或其他参数。

当前计划是学习用内存实现：服务重启后记录会丢失，也没有真实登录系统；`actor_id` 只是教学占位，生产环境必须从 JWT/session 中可信地取得当前用户。确认接口本身不会写入 Java 业务系统，已确认计划由 `/tickets/confirmations/{confirmation_id}/execute` 专门消费并执行。

| 错误码 | HTTP 状态码 | 含义 |
| --- | --- | --- |
| `TOOL_CONFIRMATION_NOT_REQUIRED` | 409 | 当前工具是只读工具，不应创建确认计划 |
| `TOOL_CONFIRMATION_NOT_FOUND` | 404 | 确认 ID 不存在 |
| `TOOL_CONFIRMATION_FORBIDDEN` | 403 | 其他操作者尝试确认该计划 |
| `TOOL_CONFIRMATION_EXPIRED` | 409 | 确认计划过期，需要重新创建 |

## 创建工单流程 `/tickets`

阶段 3 第 15 节把第 14 节的 `confirmed` 计划真正用于写操作。流程分三步：

```text
POST /tickets/plans
-> 模型提取 TicketExtraction
-> 后端转换成 CreateTicketArgs
-> 创建 create_ticket 的 pending 确认计划

POST /tools/confirmations/{confirmation_id}/confirm
-> 同一操作者确认这份固定计划

POST /tickets/confirmations/{confirmation_id}/execute
-> 读取已确认计划
-> 重新校验 CreateTicketArgs
-> 执行前再次 authorize_tool_call()
-> 使用 confirmation_id 做幂等键
-> JavaTicketClient 调用 Java mock POST /tickets
-> 校验 CreatedTicket 后返回
```

关键边界：

- `/tickets/plans` 不创建工单，只创建计划；
- `/tools/confirmations/{confirmation_id}/confirm` 只确认计划，不创建工单；
- `/tickets/confirmations/{confirmation_id}/execute` 才会调用 Java mock 服务创建工单；
- 执行接口不接收新的工单参数，只读取后端保存的确认计划；
- 当前仍是教学版，`actor_id` 和内存 store 不能直接用于生产。

相关文件：

```text
app/schemas/ticket.py
app/services/ticket_workflow_service.py
app/services/java_ticket_client.py
app/routers/tickets.py
```

## 模型工具决策 `/tool-decision`

`/tool-decision` 当前通过 `app/services/tool_decision_service.py` 调用 OpenAI-compatible 模型，并把后端允许模型看到的工具定义传给模型：

```python
tools=list_model_callable_openai_tools()
tool_choice="auto"
```

当前模型可见工具只包含启用的、只读的、不需要用户确认的工具，所以只有：

```text
query_order
```

调用链路：

```text
POST /tool-decision
-> app/routers/chat.py
-> ToolDecisionService.decide()
-> build_tool_decision_messages()
-> client.chat.completions.create(..., tools=..., tool_choice="auto")
-> extract_tool_decision()
-> authorize_tool_call(tool_name)
-> parse_tool_call_arguments(raw_arguments)
-> QueryOrderArgs.model_validate(arguments)
-> ToolDecisionResponse
```

请求示例：

```json
{
  "message": "帮我查一下订单 A1001"
}
```

如果模型决定调用工具，响应类似：

```json
{
  "decision": "call_tool",
  "reply": null,
  "tool_call": {
    "name": "query_order",
    "arguments": {
      "order_id": "A1001"
    },
    "call_id": "call_001"
  }
}
```

如果模型决定直接回答，响应类似：

```json
{
  "decision": "answer_directly",
  "reply": "请提供订单号后我再帮你查询。",
  "tool_call": null
}
```

注意：本接口当前只返回模型工具调用意图，不执行 `query_order`，也不调用 Java mock API。真正执行只读工具并把结果交回模型总结的链路由 `/tool-chat` 负责。

当前工具决策阶段新增错误：

| 错误码 | HTTP 状态码 | 含义 |
| --- | --- | --- |
| `TOOL_ARGUMENTS_INVALID_JSON` | 502 | 模型返回的工具参数不是合法 JSON 对象 |
| `TOOL_ARGUMENTS_VALIDATION_FAILED` | 502 | 模型返回的工具参数不符合 Pydantic 参数模型 |
| `TOOL_DECISION_BAD_RESPONSE` | 502 | 模型返回的工具调用结构异常 |
| `TOOL_DECISION_TOO_MANY_CALLS` | 502 | 当前阶段一次只支持一个工具调用请求 |

## 完整工具调用 `/tool-chat`

`/tool-chat` 是阶段 3 第 13 节新增的教学接口。它保留 `/tool-decision`，用于单独观察模型“是否想调用工具”；同时新增一条完整链路，用于真正执行安全的只读工具并生成用户可读回答。

调用链路：

```text
POST /tool-chat
-> app/routers/chat.py
-> ToolCallingChatService.generate_reply()
-> 第一轮 client.chat.completions.create(..., tools=..., tool_choice="auto")
-> 模型返回 query_order tool_call
-> 后端再次授权、校验参数与 call_id
-> fake_order_tool.query_order()
-> JavaOrderClient -> GET java-mock-service /orders/{order_id}
-> QueryOrderResult 校验与字段白名单映射
-> assistant tool-call message + tool message(tool_call_id, JSON content)
-> 第二轮 client.chat.completions.create(...)
-> ChatResponse.reply
```

请求示例：

```json
{
  "message": "帮我查订单 A1001 的物流"
}
```

成功响应示例：

```json
{
  "reply": "订单 A1001 已付款，商家已接单，仓库正在准备出库。"
}
```

这里的关键不是“再发一次 prompt”，而是保留协议关系：第二轮消息里先放模型第一轮的 assistant tool-call message，再放后端产生的 tool message。tool message 必须使用同一个 `tool_call_id`，并把已经校验过的工具结果序列化成 JSON 字符串。这样模型才知道这份业务数据对应哪一个工具请求。

当前为保持学习边界，只支持一轮、一个只读工具。如果第二轮模型又返回 `tool_calls`，后端不会继续自动执行，而是返回：

| 错误码 | HTTP 状态码 | 含义 |
| --- | --- | --- |
| `TOOL_CALL_ID_MISSING` | 502 | 第一轮模型工具请求没有可用于关联 tool result 的编号 |
| `TOOL_SUMMARY_UNEXPECTED_TOOL_CALL` | 502 | 第二轮模型又请求工具；多轮工具循环不属于当前小节 |

订单不存在、Java 服务超时或上游异常时，后端会直接返回已有统一错误，例如 `ORDER_NOT_FOUND`、`TOOL_TIMEOUT`、`TOOL_UPSTREAM_ERROR`；不会把失败伪装成成功结果再交给模型总结。

手动验证完整链路时，需要分别启动两个服务：

```powershell
# 终端 1：Java mock 业务服务
cd D:\wendang\java+python+ai\projects\java-mock-service
uv run uvicorn app.main:app --reload --port 8001

# 终端 2：AI 服务
cd D:\wendang\java+python+ai\projects\ai-service
uv run uvicorn app.main:app --reload --port 8000
```

再向 `POST http://127.0.0.1:8000/tool-chat` 发送上面的请求体。这个手动调用会请求本机配置的模型，可能产生费用；自动化测试不会调用真实模型或真实 Java 服务。

## 模型和工具调用测试工具

测试代码里的共享 fake 工具放在：

```text
tests/fakes.py
tests/tool_fakes.py
```

当前提供：

| 工具 | 作用 |
| --- | --- |
| `FakeOpenAICompatibleClient` | 模拟 `client.chat.completions.create(...)` 结构 |
| `FakeChatCompletions` | 模拟普通响应、流式响应、错误和调用参数记录 |
| `make_stream_chunk()` | 构造流式响应 chunk |
| `make_usage()` | 构造 token usage |
| `make_status_error()` | 构造 OpenAI SDK 风格的状态码错误 |
| `FakeOrderLookupClient` | 模拟订单查询工具依赖，记录被查询过的订单号 |
| `FakeTicketExtractor` | 模拟模型提取工单字段，避免测试真实调用 LLM |
| `FakeTicketCreator` | 模拟 Java 工单创建客户端，记录参数和幂等键 |
| `make_java_order_payload()` | 构造 Java 订单服务返回数据 |
| `make_ticket_extraction()` | 构造结构化工单字段 |
| `make_created_ticket()` | 构造 Java 工单创建返回结果 |

service 测试通过 fake client 验证模型调用参数，例如 `model`、`messages`、`stream`、`stream_options` 和 `response_format`。

工具调用测试通过 fake tool 或 fake Java client 验证业务编排；HTTP client 测试通过 `httpx.MockTransport` 模拟 Java API 响应；router/API 测试通过 FastAPI `dependency_overrides` 注入 fake service，避免测试接口时真实调用模型或真实 Java 服务。

## 阶段 2 验收

- [x] `/chat` 能调用 OpenAI-compatible 模型
- [x] `/chat` 支持多轮 `history`
- [x] `/stream-chat` 支持 SSE 流式输出
- [x] `/extract-ticket` 支持 JSON Mode 和 Pydantic 结构化校验
- [x] API key 只从本机 `.env` 或环境变量读取
- [x] timeout、rate limit、认证失败、连接失败等模型错误会映射成统一错误码
- [x] 模型调用日志记录 provider、model、耗时、token 和错误码
- [x] 日志不记录完整用户输入、完整 prompt、完整模型回复或 API key
- [x] 自动化测试使用 fake service / fake client，不真实调用模型
- [x] `uv run pytest -q` 全量通过

## CORS

本项目使用 FastAPI 的 `CORSMiddleware` 处理浏览器跨源访问。

配置入口：

```text
app/core/cors.py
```

允许来源从配置读取：

```text
CORS_ALLOWED_ORIGINS
```

默认允许：

```text
http://localhost:5173
http://127.0.0.1:5173
```

如果前端开发服务器端口不同，需要修改 `.env` 里的 `CORS_ALLOWED_ORIGINS`。

## 日志

本项目使用 Python 标准库 `logging`。

日志配置入口：

```text
app/core/logging.py
```

应用启动时会读取配置：

```text
LOG_LEVEL
```

当前 `/chat` 接口会记录一条业务日志：

```text
chat_requested message_length=...
```

模型调用成功时会记录：

```text
llm_chat_succeeded provider=... model=... elapsed_ms=... prompt_tokens=... completion_tokens=... total_tokens=...
```

模型调用失败时会记录：

```text
llm_chat_failed code=... provider=... model=... status_code=... elapsed_ms=...
```

流式模型调用成功时会记录：

```text
llm_stream_chat_succeeded provider=... model=... elapsed_ms=... chunks=... content_chunks=... prompt_tokens=... completion_tokens=... total_tokens=...
```

流式模型调用失败时会记录：

```text
llm_stream_chat_failed code=... provider=... model=... status_code=... elapsed_ms=... chunks=... content_chunks=...
```

日志格式会自动带上当前请求的 `trace_id`：

```text
trace_id=...
```

注意：日志只记录消息长度、模型名、服务商、耗时、token 用量和错误码等元信息，不记录完整用户输入、完整 `history`、完整 prompt、完整模型回复或 API key，避免把敏感内容写入日志。

## 请求追踪

本项目使用 `trace_id` 追踪一次 HTTP 请求。

相关文件：

```text
app/core/trace.py
app/middleware/tracing.py
app/services/java_order_client.py
app/services/java_ticket_client.py
```

每个请求都会经过 trace middleware：

```text
请求进入 -> 设置 trace_id -> 执行路由 -> 响应头返回 X-Trace-Id -> 清理 trace_id
```

如果客户端传入：

```text
X-Trace-Id
```

服务端会复用它。

如果客户端没有传，服务端会生成新的 `trace_id`。

阶段 3 第 16 节开始，Python AI 服务调用 Java mock 时也会把当前 `trace_id` 放进出站请求头：

```text
X-Trace-Id: 当前请求的 trace_id
```

这样未来 Java 业务服务也记录 trace_id 时，就可以用同一个编号关联 Python 和 Java 两边日志。没有真实请求上下文时，不会把占位符 `-` 当作 trace_id 传给下游。

## 统一异常处理

本项目使用统一错误响应格式：

```json
{
  "code": "VALIDATION_ERROR",
  "message": "请求参数校验失败",
  "trace_id": "..."
}
```

如果错误有结构化细节，会额外返回：

```json
{
  "details": []
}
```

相关文件：

```text
app/schemas/error.py
app/core/exceptions.py
app/core/exception_handlers.py
```

当前已统一处理：

| 类型 | code |
| --- | --- |
| 404 | `NOT_FOUND` |
| 405 | `METHOD_NOT_ALLOWED` |
| 参数校验错误 | `VALIDATION_ERROR` |
| 业务异常 | 自定义业务错误码 |
| 未知异常 | `INTERNAL_SERVER_ERROR` |

## 当前接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务健康检查 |
| POST | `/chat` | 聊天接口，调用 OpenAI-compatible 模型 |
| POST | `/langchain-chat` | LangChain ChatModel 聊天接口，用于对比原生 SDK 调用 |
| POST | `/stream-chat` | 流式聊天接口，调用 OpenAI-compatible 模型并返回 SSE |
| POST | `/extract-ticket` | 结构化工单字段抽取接口，调用 OpenAI-compatible 模型并用 Pydantic 校验 |
| POST | `/langchain-extract-ticket` | LangChain 结构化工单字段抽取接口，使用 `with_structured_output()` |
| POST | `/tool-decision` | 模型工具调用决策接口，返回直接回答或工具调用意图 |
| POST | `/tool-chat` | 完整工具调用接口：执行只读工具后将结果交给模型总结 |
| POST | `/tools/query-order` | 订单查询工具接口，通过 Java mock API 查询订单 |
| GET | `/tools/langchain` | 查看 LangChain Tool 元数据 |
| POST | `/tools/langchain/query-order` | 手动调用包装后的 LangChain `query_order` Tool |
| POST | `/tools/confirmations` | 创建绑定工具、参数和操作者的待确认计划，不执行工具 |
| POST | `/tools/confirmations/{confirmation_id}/confirm` | 确认已有计划，不执行工具 |
| POST | `/tickets/plans` | 从用户问题提取工单字段，并创建待确认的创建工单计划 |
| POST | `/tickets/confirmations/{confirmation_id}/execute` | 执行已确认的创建工单计划，调用 Java mock API |

## 当前模型

| 模型 | 路径 | 说明 |
| --- | --- | --- |
| `ChatRequest` | `app/schemas/chat.py` | 聊天请求体，要求 `message` 是非空字符串，可选 `history` 做多轮上下文 |
| `ChatResponse` | `app/schemas/chat.py` | 聊天响应体，要求 `reply` 是非空字符串 |
| `ChatMessageRole` | `app/schemas/chat.py` | 聊天消息角色，只允许 `system`、`user`、`assistant` |
| `ChatMessage` | `app/schemas/chat.py` | 聊天消息模型，包含 `role` 和 `content` |
| `StructuredOutputRequest` | `app/schemas/structured.py` | 结构化输出请求体，要求 `message` 是非空字符串 |
| `TicketExtraction` | `app/schemas/structured.py` | 工单字段抽取结果，包含 `intent`、`order_id`、`summary`、`urgency`、`need_human_review` |
| `StructuredOutputResponse` | `app/schemas/structured.py` | 结构化输出响应体，包裹 `TicketExtraction` |
| `CreateTicketArgs` | `app/schemas/ticket.py` | 后端拥有的创建工单业务命令 |
| `CreatedTicket` | `app/schemas/ticket.py` | Java mock 服务返回后经 Python 校验的工单结果 |
| `TicketPlanRequest` | `app/schemas/ticket.py` | 创建工单计划请求，包含操作者和用户自然语言问题 |
| `TicketPlanResponse` | `app/schemas/ticket.py` | 返回模型提取结果和待确认计划 |
| `TicketExecutionResponse` | `app/schemas/ticket.py` | 返回确认 ID 和创建出的工单 |
| `ToolAccessLevel` | `app/schemas/tool.py` | 工具风险等级枚举，当前包括 `read`、`write`、`sensitive` |
| `ToolDefinition` | `app/schemas/tool.py` | 后端工具定义模型，包含工具名、描述、风险等级、是否启用和是否需要确认 |
| `QueryOrderArgs` | `app/schemas/tool.py` | `query_order` 工具参数模型，要求 `order_id` 非空且格式合法 |
| `OrderStatus` | `app/schemas/tool.py` | 订单状态枚举，如 `waiting_shipment`、`shipped`、`delivered` |
| `PaymentStatus` | `app/schemas/tool.py` | 支付状态枚举，如 `unpaid`、`paid`、`refunded` |
| `QueryOrderResult` | `app/schemas/tool.py` | 订单查询工具结果，包含订单状态、支付状态、物流说明和是否可创建工单 |
| `QueryOrderResponse` | `app/schemas/tool.py` | 订单查询接口响应体，包裹 `QueryOrderResult` |
| `ToolDecisionType` | `app/schemas/tool_decision.py` | 工具决策枚举，当前包含 `answer_directly` 和 `call_tool` |
| `ToolCallCandidate` | `app/schemas/tool_decision.py` | 模型请求的工具调用候选，包含工具名、参数和可选 `call_id` |
| `ToolDecisionResponse` | `app/schemas/tool_decision.py` | `/tool-decision` 响应体，表示模型直接回答或请求工具 |
| `ErrorResponse` | `app/schemas/error.py` | 统一错误响应体，包含 `code`、`message`、`trace_id` 和可选 `details` |

## 阶段 1 验收

- [x] 服务可以启动
- [x] `/health` 可以访问
- [x] `/chat` 可以接收 JSON 请求体
- [x] 请求和响应使用 Pydantic 模型
- [x] 配置从 `.env` 或环境变量读取
- [x] 日志可以通过 `LOG_LEVEL` 控制
- [x] 每个请求都有 `X-Trace-Id`
- [x] 错误响应格式统一
- [x] CORS 允许配置的前端来源
- [x] 自动化测试通过

## 阶段 3 收尾状态

阶段 3 已经完成 LangChain + Java 工具调用基础，为后续智能工单 Agent 调用 Java 业务服务打好了基础。

当前阶段 3 的核心目标：

- 理解 Tool Calling 不是模型直接执行代码，而是模型返回工具名和参数，由后端决定是否执行。
- 理解 AI 不能绕过 Java 后端直接操作业务系统，模型输出必须当成不可信输入处理。
- 理解工具参数如何用 JSON Schema 描述，并用 Pydantic 在后端校验模型返回的 arguments。
- 理解结构化输出和 Tool Calling 的区别：前者把自然语言整理成固定格式数据，后者让模型提出调用外部工具的请求。
- 用 fake tool 先模拟订单查询，避免一开始就引入复杂业务服务。
- 理解工具返回结果、Java API 响应和第三方接口返回也要先用 Pydantic 校验。
- 理解工具调用失败时要把 timeout、404、上游 500 映射成统一、安全、可测试的项目错误。
- 理解工具调用必须经过后端白名单、启用状态、风险等级和用户确认守卫，不能由模型自行决定权限。
- 理解重复工具调用要通过 `Idempotency-Key` 和参数指纹避免重复产生业务效果。
- 已用 FastAPI 写一个 Java mock 业务服务，模拟后续 Spring Boot 接口。
- 当前已经让 Python AI 服务调用 Java mock API，并处理超时、404、500、权限和幂等。
- 当前已经完成一个只读工具的执行、tool message 回传和第二轮模型总结。
- 当前已经完成写操作前的确认计划、操作者绑定、参数绑定、过期和确认幂等。
- 当前已经完成创建工单流程：提取字段、使用已确认计划、调用 Java mock API，并用 `confirmation_id` 做幂等保护。
- 当前已经完成工具调用日志和 `trace_id` 串联，让模型、工具、确认计划和 Java API 调用更容易排查。
- 当前已经完成工具调用测试：fake Java API / fake tool，整理了模型 fake、工具 fake、`MockTransport` 和 `dependency_overrides` 的分层测试策略。
- 当前已经完成 LangChain 的框架定位学习，明确 LangChain 是 AI 编排封装层，不是业务权限、安全、幂等和校验边界。
- 当前已经完成 LangChain ChatModel 基础，新增 `/langchain-chat` 对比原生 SDK 调用和 LangChain `model.invoke()` 调用。
- 当前已经完成 LangChain Tool 基础，把已有 `query_order` 包装成 `StructuredTool`，并保留项目工具注册表、权限和参数校验边界。
- 当前已经完成 LangChain 结构化输出，新增 `/langchain-extract-ticket` 对比原生 JSON Mode + Pydantic 和 LangChain `with_structured_output()`。
- 当前已经完成阶段 3 项目整理，形成 Tool Calling、Java mock API、确认机制、trace_id、分层测试和 LangChain 封装的完整知识地图。
- 当前正在阶段 4 Milvus 对比学习，已跑通同一批知识文档写入 Milvus、按 metadata 过滤检索，并为常用过滤字段创建 scalar index。
- 后续会把业务工具调用能力和企业知识检索能力一起作为智能工单 Agent 的基础。
