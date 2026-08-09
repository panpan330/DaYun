# Java Business Service

这个项目是阶段 7 新增的真实 Java Spring Boot 业务服务骨架。

它的定位不是 AI 服务，也不是模型服务，而是：

```text
AI Agent 可以安全、稳定、可追踪调用的 Java 业务系统。
```

当前阶段 7 第 8 节已完成到 MySQL + Redis + MyBatis 传统结构真实化，并补强 AI 场景下的内部鉴权和用户身份传递：

```text
Spring Boot 启动入口
internal API Controller
统一 ApiResponse
传统 controller / service / mapper / entity / dto 目录结构
内部调用 Header 校验
内部调用方 allowed-caller 配置化
X-Tenant-Id 必传
trace_id / caller / user_id / tenant_id 基础格式校验
trace_id 透传
创建工单幂等雏形
基础契约测试
MySQL 表设计草案
订单查询 MyBatis + MySQL 读链路
创建工单 MyBatis + MySQL 写链路
订单查询 Redis 缓存
创建工单 Redis 幂等缓存
internal 工具接口 Redis 限流
```

后续阶段会继续接入：

```text
真实用户表和完整权限体系
Python AI 服务适配
Docker Compose
```

MySQL 设计文档：

```text
../../docs/java-business-database-design.md
```

当前 MySQL 落地状态：

```text
orders 表已通过 schema.sql / data.sql 初始化。
GET /internal/orders/{order_id} 已通过 OrderMapper + OrderMapper.xml 从 MySQL 读取。
tickets 表和 ticket_events 表已通过 schema.sql 初始化。
POST /internal/tickets 已通过 TicketMapper + TicketMapper.xml 写入 MySQL。
创建工单使用 MySQL 唯一索引、request_fingerprint 和事务处理幂等与事件写入。
```

当前 Redis 接入状态：

```text
默认连接 VMware Ubuntu Docker Redis：192.168.88.10:6379。
GET /internal/orders/{order_id} 已接入 Redis read-through cache。
POST /internal/tickets 已接入 Redis 幂等缓存，但仍以 MySQL 唯一索引兜底。
internal 工具接口已接入 Redis fixed window 限流。
测试环境默认 app.redis.enabled=false，不依赖真实 Redis。
```

当前传统结构：

```text
controller / service / service.impl / mapper / entity / dto / config / exception / common 已落地。
数据访问层已从 JdbcTemplate 切换到 MyBatis Mapper + XML。
重构后仍保留 AI Agent 调用边界：DTO 白名单、权限、幂等、trace_id、错误码和 internal token。
InternalRequestResolver 已统一校验 internal token、allowed caller、真实用户身份、租户身份和基础 header 格式。
```

## 运行

本地运行前先确认 Windows MySQL 已启动，并创建数据库：

```powershell
$env:MYSQL_PWD = "你的 MySQL 密码"
mysql -u root -h 127.0.0.1 -P 3306 -e "CREATE DATABASE IF NOT EXISTS ai_business DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
```

设置 Java 服务读取 MySQL 密码：

```powershell
$env:JAVA_BUSINESS_DB_PASSWORD = "你的 MySQL 密码"
```

设置 internal 接口的内部鉴权配置：

```powershell
$env:JAVA_BUSINESS_INTERNAL_TOKEN = "local-dev-internal-token"
$env:JAVA_BUSINESS_INTERNAL_ALLOWED_CALLER = "ai-service"
```

如果需要启用 Redis，先确认 VMware Ubuntu 虚拟机里的 Redis 容器已启动，并且 Windows 可以连通：

```powershell
Test-NetConnection 192.168.88.10 -Port 6379
```

Redis 连接配置可以通过环境变量覆盖：

```powershell
$env:JAVA_BUSINESS_REDIS_ENABLED = "true"
$env:JAVA_BUSINESS_REDIS_HOST = "192.168.88.10"
$env:JAVA_BUSINESS_REDIS_PORT = "6379"
```

如果虚拟机没开，但只想先运行 Java 服务，可以临时关闭 Redis：

```powershell
$env:JAVA_BUSINESS_REDIS_ENABLED = "false"
```

```powershell
mvn spring-boot:run
```

默认端口：

```text
8002
```

健康检查：

```text
GET http://127.0.0.1:8002/health
GET http://127.0.0.1:8002/ready
```

## 测试

```powershell
mvn test
```

测试环境使用 H2 内存数据库，但仍然走 `OrderMapper`、`TicketMapper` 和 MyBatis XML 链路。
测试环境默认关闭真实 Redis，使用 NoOp cache / NoOp rate limiter，避免 `mvn test` 依赖 VMware 虚拟机。

## 当前 internal 接口

```text
GET /internal/orders/{order_id}
POST /internal/tickets
```

这两个接口按 [../../docs/java-ai-api-contract.md](../../docs/java-ai-api-contract.md) 的方向设计。

调用时必须携带 `X-Trace-Id`、`X-Caller`、`X-User-Id`、`X-Tenant-Id` 和 `X-Internal-Token`。

后续 MySQL 表结构按 [../../docs/java-business-database-design.md](../../docs/java-business-database-design.md) 落地。
