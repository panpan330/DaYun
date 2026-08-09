# 本地运行与演示说明

本文档面向阶段 11 的完整项目化版本，目标是把前端、Java business service、Python AI service、MySQL、Redis、Qdrant 和真实模型配置按稳定顺序跑起来。

当前核心服务：

```text
projects/customer-service-console   Vue3 + Element Plus 前端
projects/java-business-service      Spring Boot + MyBatis + MySQL/Redis 业务服务
projects/ai-service                 FastAPI + RAG/Agent/LLM AI 服务
Windows MySQL                       业务数据
远程/虚拟机 Redis                 登录、限流、缓存、幂等
远程/虚拟机 Qdrant                RAG 向量检索
```

## 1. 本地端口

| 服务 | 地址 |
| --- | --- |
| 前端控制台 | `http://127.0.0.1:5173` |
| Java business service | `http://127.0.0.1:18004` |
| Python AI service | `http://127.0.0.1:8000` |
| MySQL | `127.0.0.1:3306` |
| Redis | `<INFRA_HOST>:6379` |
| Qdrant | `http://<INFRA_HOST>:6333` |

如果基础设施服务器 IP 变化，先在其上执行：

```bash
hostname -I
```

然后同步修改：

```text
projects/ai-service/.env
projects/java-business-service 的运行环境变量
```

## 2. 启动前准备

### 2.1 Windows MySQL

需要有数据库：

```text
ai_business
```

本地默认配置：

```text
username=root
password=root
```

如果数据库不存在，先在 MySQL 中执行：

```sql
CREATE DATABASE IF NOT EXISTS ai_business
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;
```

Java 服务启动时会根据 `schema.sql` 和 `data.sql` 初始化表结构和演示数据。

### 2.2 Redis（远程/虚拟机）

在 Ubuntu 中启动 Redis 容器：

```bash
docker start redis-server
docker ps --filter name=redis
```

Windows 验证端口：

```powershell
Test-NetConnection <INFRA_HOST> -Port 6379
```

如果暂时不想用 Redis，可以启动 Java 服务时设置：

```powershell
$env:JAVA_BUSINESS_REDIS_ENABLED = "false"
```

### 2.3 Qdrant（远程/虚拟机）

RAG 真实链路需要 Qdrant。启动命令：

```bash
docker start qdrant
docker ps --filter name=qdrant
curl http://localhost:6333
```

Windows 验证：

```powershell
curl.exe http://<INFRA_HOST>:6333
curl.exe http://<INFRA_HOST>:6333/collections
```

## 3. 配置文件

### 3.1 Python AI service

进入目录：

```powershell
cd D:\wendang\java+python+ai\projects\ai-service
```

如果没有 `.env`，复制示例：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

阶段 11 推荐关键配置：

```text
TICKET_AGENT_MODEL_MODE=real_llm
JAVA_BUSINESS_SERVICE_BASE_URL=http://127.0.0.1:18004
JAVA_BUSINESS_INTERNAL_TOKEN=local-dev-internal-token
JAVA_BUSINESS_INTERNAL_CALLER=ai-service
JAVA_BUSINESS_DEFAULT_USER_ID=U1001
JAVA_BUSINESS_DEFAULT_TENANT_ID=default

QDRANT_BASE_URL=http://<INFRA_HOST>:6333
QDRANT_COLLECTION_NAME=learning_rag_chunks_v4_1024
QDRANT_VECTOR_SIZE=1024

LLM_MODEL=qwen3.7-plus
LLM_BASE_URL=https://你的-workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=你的真实Key

EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BASE_URL=https://你的-workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=你的真实Key
EMBEDDING_DIMENSION=1024

RERANK_MODEL=qwen3-rerank
RERANK_BASE_URL=https://你的-workspace-id.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_API_KEY=你的真实Key
```

`.env` 不提交 GitHub。`.env.example` 只放占位符。

### 3.2 Java business service

参考文件：

```text
projects/java-business-service/.env.example
```

注意：Spring Boot 不会自动读取 `.env.example` 或 `.env`。你需要在 IDEA Run Configuration 里配置环境变量，或者用 PowerShell `$env:` 设置。

本地 PowerShell 示例：

```powershell
cd D:\wendang\java+python+ai\projects\java-business-service
$env:JAVA_BUSINESS_DB_PASSWORD = "root"
$env:JAVA_BUSINESS_REDIS_ENABLED = "true"
$env:JAVA_BUSINESS_REDIS_HOST = "<INFRA_HOST>"
$env:JAVA_BUSINESS_REDIS_PORT = "6379"
$env:JAVA_BUSINESS_INTERNAL_TOKEN = "local-dev-internal-token"
$env:JAVA_BUSINESS_INTERNAL_ALLOWED_CALLER = "ai-service"
```

### 3.3 前端控制台

进入目录：

```powershell
cd D:\wendang\java+python+ai\projects\customer-service-console
```

如果没有 `.env`，复制示例：

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

默认配置：

```text
VITE_JAVA_API_BASE_URL=http://127.0.0.1:18004
VITE_AI_API_BASE_URL=http://127.0.0.1:8000
```

## 4. 启动顺序

### 4.1 启动 Java business service

```powershell
cd D:\wendang\java+python+ai\projects\java-business-service
mvn spring-boot:run
```

健康检查：

```powershell
curl.exe http://127.0.0.1:18004/health
```

预期：

```json
{"service":"java-business-service","status":"ok"}
```

### 4.2 启动 Python AI service

```powershell
cd D:\wendang\java+python+ai\projects\ai-service
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/ready
```

如果 `TICKET_AGENT_MODEL_MODE=real_llm` 且没有配置模型 Key，`/ready` 返回 503 是正常的，表示真实模型配置不完整。

### 4.3 启动前端控制台

```powershell
cd D:\wendang\java+python+ai\projects\customer-service-console
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

## 5. 演示账号

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `customer` | `123456` | 普通用户 |
| `customer2` | `123456` | 普通用户 |
| `agent` | `123456` | 客服 |
| `supervisor` | `123456` | 主管 |
| `admin` | `123456` | 管理员 |

推荐演示顺序：

```text
agent 登录
-> 查看运营概览
-> 查看订单列表
-> 查看工单工作台
-> 查看知识库管理
-> 查看 AI 评估页面
-> 打开 AI 客服页演示 RAG / Agent 能力
```

## 6. 关键接口验证

### 6.1 Java public API

登录：

```powershell
$loginBody = @{ username="agent"; password="123456" } | ConvertTo-Json -Compress
$login = curl.exe -s -X POST "http://127.0.0.1:18004/api/auth/login" `
  -H "Content-Type: application/json" `
  --data-raw $loginBody | ConvertFrom-Json
$token = $login.data.token
```

查工单列表：

```powershell
curl.exe -s "http://127.0.0.1:18004/api/tickets" `
  -H "Authorization: Bearer $token" | ConvertFrom-Json
```

查工单详情：

```powershell
curl.exe -s "http://127.0.0.1:18004/api/tickets/T-DEMO-1001" `
  -H "Authorization: Bearer $token" | ConvertFrom-Json
```

更新工单状态：

```powershell
$body = @{ target_status="in_progress"; note="开始跟进物流问题" } | ConvertTo-Json -Compress
curl.exe -s -X PATCH "http://127.0.0.1:18004/api/tickets/T-DEMO-1001/status" `
  -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  --data-raw $body | ConvertFrom-Json
```

### 6.2 Python AI API

AI 评估看板：

```powershell
curl.exe -s "http://127.0.0.1:8000/api/ai/evaluation/overview" | ConvertFrom-Json
```

RAG 问答：

```powershell
$body = @{ query="退款多久到账"; candidate_count=20; top_n=5 } | ConvertTo-Json -Compress
curl.exe -s -X POST "http://127.0.0.1:8000/api/ai/rag/ask" `
  -H "Content-Type: application/json" `
  --data-raw $body | ConvertFrom-Json
```

AI 对话：

```powershell
$body = @{ message="帮我查一下订单 A1001"; history=@() } | ConvertTo-Json -Depth 5 -Compress
curl.exe -s -X POST "http://127.0.0.1:8000/api/ai/chat" `
  -H "Content-Type: application/json" `
  --data-raw $body | ConvertFrom-Json
```

## 7. 测试命令

Java：

```powershell
cd D:\wendang\java+python+ai\projects\java-business-service
mvn test
```

Python：

```powershell
cd D:\wendang\java+python+ai\projects\ai-service
uv run pytest -q
```

前端：

```powershell
cd D:\wendang\java+python+ai\projects\customer-service-console
npm run build
```

## 8. 常见问题

### 8.1 PowerShell 里中文像乱码

先怀疑 PowerShell 输出编码或字体显示问题。不要直接大范围修改项目文件。

优先用这些方式确认：

```text
浏览器页面
VS Code UTF-8 打开文件
接口 JSON 原始响应
```

### 8.2 `curl` 弹出安全提示

PowerShell 里的 `curl` 可能是 `Invoke-WebRequest` 的别名。需要真实 curl 时写：

```powershell
curl.exe
```

### 8.3 `curl.exe` JSON 被拆开

PowerShell 对引号比较敏感。推荐先构造 `$body`：

```powershell
$body = @{ key="value" } | ConvertTo-Json -Compress
curl.exe -X POST "http://127.0.0.1:8000/example" `
  -H "Content-Type: application/json" `
  --data-raw $body
```

### 8.4 Redis 连接失败

先检查虚拟机：

```bash
docker ps --filter name=redis
hostname -I
```

再检查 Windows：

```powershell
Test-NetConnection <INFRA_HOST> -Port 6379
```

如果只是临时演示 Java + MySQL 主线，可设置：

```powershell
$env:JAVA_BUSINESS_REDIS_ENABLED = "false"
```

### 8.5 Qdrant 连接失败

先检查虚拟机：

```bash
docker ps --filter name=qdrant
curl http://localhost:6333
hostname -I
```

再检查 Windows：

```powershell
curl.exe http://<INFRA_HOST>:6333
```

确认 `projects/ai-service/.env` 中的 `QDRANT_BASE_URL` 和虚拟机 IP 一致。

### 8.6 Python 报 `ModuleNotFoundError: No module named 'app'`

通常是运行目录不对。先进入 `ai-service` 项目根目录：

```powershell
cd D:\wendang\java+python+ai\projects\ai-service
uv run python scripts\xxx.py
```

如果某个脚本仍然报错，再临时设置：

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

## 9. 关闭服务

在启动服务的终端中按：

```text
Ctrl + C
```

Docker 容器可按需停止：

```bash
docker stop qdrant
docker stop redis-server
```
