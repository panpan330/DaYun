# Java AI API 契约草案

本文档记录阶段 7 中 Python AI 服务调用真实 Java Spring Boot 业务服务时的接口契约草案。

当前目标：

```text
记录 Python AI 服务调用真实 Java Spring Boot 业务服务时必须遵守的接口契约。
阶段 7 已完成真实 Java business service 底座，后续 Python 运行时链路迁移应以本文档和共享契约文件为准。
```

本契约暂时覆盖两个核心工具接口：

```text
GET /internal/orders/{order_id}
POST /internal/tickets
```

## 1. 契约原则

### 1.1 核心边界

```text
模型提出意图，后端执行动作。
```

Python AI 服务负责：

```text
理解用户自然语言
执行 RAG
管理 Tool Calling
做工具参数第一层校验
处理用户确认
调用 Java HTTP API
把工具结果交给模型总结
```

Java 业务服务负责：

```text
用户身份校验
内部调用鉴权
权限判断
业务规则校验
事务
幂等
MySQL 持久化
Redis 缓存/限流/短期状态
机器可读错误码
审计日志
```

### 1.2 字段暴露原则

```text
Java Entity 不直接暴露给 Python AI 服务。
Java Response DTO 只返回工具所需字段。
Python 再做 Pydantic 校验和模型可见字段白名单。
模型只看到完成回答所需的最小信息。
```

## 2. 通用 Header

| Header | 必填 | 适用范围 | 说明 |
| --- | --- | --- | --- |
| `X-Trace-Id` | 是 | 所有接口 | 串联 Python + Java + MySQL/Redis 日志 |
| `X-Caller` | 是 | 所有接口 | 调用方，例如 `ai-service` |
| `X-User-Id` | 是 | 业务接口 | 当前真实用户，不由模型生成 |
| `X-Tenant-Id` | 是 | 业务接口 | 当前租户/业务域，暂无多租户时显式传 `default` |
| `X-Internal-Token` | 是 | 所有 internal 接口 | 内部服务鉴权 |
| `Idempotency-Key` | 写接口必填 | 创建/修改类接口 | 防重复写入 |

## 3. 统一响应结构

成功：

```json
{
  "success": true,
  "code": "OK",
  "message": "OK",
  "data": {},
  "trace_id": "4f7d..."
}
```

失败：

```json
{
  "success": false,
  "code": "ORDER_NOT_FOUND",
  "message": "订单不存在，请确认订单号是否正确。",
  "data": null,
  "trace_id": "4f7d..."
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `success` | boolean | 业务是否成功 |
| `code` | string | 机器可读业务码 |
| `message` | string | 简短人类可读说明 |
| `data` | object/null | 成功时业务数据，失败时通常为 null |
| `trace_id` | string | 链路追踪 ID |

约定：

```text
不要把所有错误都返回 HTTP 200。
HTTP 状态码表达协议/错误大类。
code 表达具体业务原因。
```

## 4. 订单查询接口

### 4.1 基本信息

```text
GET /internal/orders/{order_id}
```

工具类型：

```text
read
```

用途：

```text
给 Python AI 服务查询当前用户有权查看的订单摘要。
```

### 4.2 请求 Header

```text
X-Trace-Id: 4f7d...
X-Caller: ai-service
X-User-Id: U1001
X-Tenant-Id: default
X-Internal-Token: ***
```

### 4.3 Path 参数

| 参数 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `order_id` | string | 是 | 1-64 位，字母/数字/下划线/短横线 | 订单号 |

### 4.4 成功响应

HTTP：

```text
200 OK
```

Body：

```json
{
  "success": true,
  "code": "OK",
  "message": "OK",
  "data": {
    "order_id": "A1001",
    "order_status": "shipped",
    "payment_status": "paid",
    "logistics_message": "订单已发货，预计 2 天内送达。",
    "latest_event": "包裹已离开发货仓。",
    "can_create_ticket": true,
    "user_visible_summary": "订单已发货，正在运输中。"
  },
  "trace_id": "4f7d..."
}
```

### 4.5 `OrderToolView` 字段

| 字段 | 类型 | 必填 | 是否可给模型 | 说明 |
| --- | --- | --- | --- | --- |
| `order_id` | string | 是 | 是 | 订单号 |
| `order_status` | string | 是 | 是 | 订单状态 |
| `payment_status` | string | 是 | 可选 | 支付状态，按场景决定是否给模型 |
| `logistics_message` | string | 是 | 是 | 物流摘要 |
| `latest_event` | string | 是 | 是 | 最新订单事件 |
| `can_create_ticket` | boolean | 是 | 是 | 是否适合创建客服工单 |
| `user_visible_summary` | string | 是 | 是 | 用户可见摘要 |

### 4.6 不应返回给模型的字段

```text
用户手机号完整值
收货人完整姓名
完整收货地址
支付流水号
内部成本
供应商信息
内部风控字段
内部客服备注
数据库自增主键
租户内部配置
```

### 4.7 错误响应

| HTTP | code | 含义 | Python/Agent 处理 |
| --- | --- | --- | --- |
| 422 | `ORDER_ID_INVALID` | 订单号格式不合法 | 让用户重新提供订单号 |
| 401 | `INTERNAL_AUTH_FAILED` | 内部鉴权失败 | 记录错误，不向用户暴露内部细节 |
| 403 | `ORDER_ACCESS_DENIED` | 当前用户无权查看订单 | 告知用户无权查看 |
| 404 | `ORDER_NOT_FOUND` | 订单不存在 | 让用户确认订单号 |
| 504 | `JAVA_SERVICE_TIMEOUT` | Java 或数据库超时 | 提示稍后重试或转人工 |
| 503 | `JAVA_SERVICE_UNAVAILABLE` | 服务不可用 | 降级或转人工 |

## 5. 创建工单接口

### 5.1 基本信息

```text
POST /internal/tickets
```

工具类型：

```text
write
```

用途：

```text
在用户确认后，由 Python AI 服务请求 Java 后端创建客服工单。
```

### 5.2 请求 Header

```text
X-Trace-Id: 4f7d...
X-Caller: ai-service
X-User-Id: U1001
X-Tenant-Id: default
X-Internal-Token: ***
Idempotency-Key: ticket-create-4f7d-A1001
```

说明：

```text
X-User-Id 表示真实用户身份，不建议从请求 body 传 requester_id。
X-Tenant-Id 表示当前租户/业务域，AI 服务即使暂时只有默认租户，也应该显式传 `default`，避免 Java 服务在权限兜底时缺少边界。
Idempotency-Key 是写操作必填项。
```

### 5.3 请求 Body

```json
{
  "title": "物流太慢",
  "description": "用户反馈 A1001 订单物流长时间未更新，希望客服跟进。",
  "category": "logistics",
  "priority": "normal",
  "related_order_id": "A1001",
  "source": "ai_agent",
  "confirmation_id": "9f4d0b2f5b0c4f2a9d6c8b1e0a3f7c11"
}
```

### 5.4 `CreateTicketCommand` 字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `title` | string | 是 | 工单标题，1-200 字符 |
| `description` | string | 是 | 工单描述，1-1000 字符 |
| `category` | string | 是 | 工单类型 |
| `priority` | string | 是 | 优先级 |
| `related_order_id` | string/null | 否 | 关联订单号 |
| `source` | string | 是 | 固定建议为 `ai_agent` |
| `confirmation_id` | string | 是 | Python AI 服务侧用户确认记录 |

`category` 建议枚举：

```text
refund
order_query
logistics
complaint
policy_gap
```

`priority` 建议枚举：

```text
low
normal
high
```

### 5.5 成功响应

HTTP：

```text
201 Created
```

Body：

```json
{
  "success": true,
  "code": "OK",
  "message": "OK",
  "data": {
    "ticket_id": "T1001",
    "ticket_status": "created",
    "title": "物流太慢",
    "category": "logistics",
    "priority": "normal",
    "related_order_id": "A1001",
    "created_at": "2026-07-27T10:30:00Z",
    "user_visible_summary": "工单已创建，客服会继续跟进物流问题。"
  },
  "trace_id": "4f7d..."
}
```

### 5.6 `TicketToolView` 字段

| 字段 | 类型 | 必填 | 是否可给模型 | 说明 |
| --- | --- | --- | --- | --- |
| `ticket_id` | string | 是 | 是 | 工单号 |
| `ticket_status` | string | 是 | 是 | 工单状态 |
| `title` | string | 是 | 是 | 工单标题 |
| `category` | string | 是 | 是 | 工单类型 |
| `priority` | string | 是 | 是 | 优先级 |
| `related_order_id` | string/null | 否 | 是 | 关联订单号 |
| `created_at` | string | 是 | 可选 | 创建时间 |
| `user_visible_summary` | string | 是 | 是 | 用户可见摘要 |

### 5.7 错误响应

| HTTP | code | 含义 | Python/Agent 处理 |
| --- | --- | --- | --- |
| 400 | `IDEMPOTENCY_KEY_REQUIRED` | 缺少幂等键 | 阻断执行，记录系统错误 |
| 409 | `IDEMPOTENCY_KEY_CONFLICT` | 同一幂等键参数冲突 | 阻断执行，提示系统异常 |
| 422 | `TICKET_REQUEST_INVALID` | 工单参数不合法 | 让用户补充或修正 |
| 401 | `INTERNAL_AUTH_FAILED` | 内部鉴权失败 | 记录错误，不暴露细节 |
| 403 | `ORDER_ACCESS_DENIED` | 无权基于该订单创建工单 | 告知无权操作 |
| 404 | `ORDER_NOT_FOUND` | 关联订单不存在 | 让用户确认订单号 |
| 409 | `TICKET_ALREADY_EXISTS` | 已存在类似工单 | 告知已有工单 |
| 409 | `ORDER_NOT_SUPPORT_TICKET` | 订单不支持创建该类工单 | 解释业务限制 |
| 504 | `JAVA_SERVICE_TIMEOUT` | 创建工单超时 | 基于幂等键重试或查询结果 |
| 503 | `JAVA_SERVICE_UNAVAILABLE` | 服务不可用 | 提示稍后再试或转人工 |

## 6. Python 错误映射建议

| Java code | Python/Agent 行为 |
| --- | --- |
| `ORDER_ID_INVALID` | 让用户重新提供订单号 |
| `ORDER_NOT_FOUND` | 告诉用户订单不存在或订单号可能有误 |
| `ORDER_ACCESS_DENIED` | 拒绝泄露订单信息 |
| `TICKET_REQUEST_INVALID` | 追问缺失或错误字段 |
| `TICKET_ALREADY_EXISTS` | 告诉用户已有工单，不重复创建 |
| `IDEMPOTENCY_KEY_CONFLICT` | 记录系统异常，不让模型编造成功 |
| `JAVA_SERVICE_TIMEOUT` | 可重试、降级或转人工 |
| `JAVA_SERVICE_UNAVAILABLE` | 降级或转人工 |

阶段 7 第 9 节补充了 Python AI 服务侧的安全映射原则：

| Java code | Python 对外 code | 用户侧安全提示方向 |
| --- | --- | --- |
| `ORDER_NOT_FOUND` | `ORDER_NOT_FOUND` | 告诉用户订单不存在，请确认订单号 |
| `ORDER_ACCESS_DENIED` | `ORDER_ACCESS_DENIED` | 告诉用户当前账号无权查看或操作该订单 |
| `ORDER_NOT_SUPPORT_TICKET` | `ORDER_NOT_SUPPORT_TICKET` | 告诉用户当前订单暂不支持创建这类工单 |
| `IDEMPOTENCY_KEY_CONFLICT` | `IDEMPOTENCY_KEY_CONFLICT` | 告诉用户本次提交和已确认的工单请求不一致，需要重新确认 |
| `INTERNAL_AUTH_FAILED` | `TOOL_UPSTREAM_ERROR` | 不暴露内部鉴权，提示业务服务暂时不可用 |
| `JAVA_SERVICE_ERROR` | `TOOL_UPSTREAM_ERROR` | 不暴露内部异常，提示业务服务暂时不可用 |
| `IDEMPOTENCY_KEY_REQUIRED` / `IDEMPOTENCY_KEY_INVALID` | `TICKET_UPSTREAM_REJECTED` | 视为 Python/Java 写接口契约问题，不让用户处理幂等键 |

原则：

```text
Java code 是机器语义，不等于用户话术。
Java message 不默认直接展示。
Python AI 服务必须先把 Java 错误响应映射成安全 AppException。
模型不能自由解释内部错误原因。
```

## 6.1 trace_id 串联约定

阶段 7 第 10 节补充了 Python + Java 的最小跨服务追踪约定：

| 位置 | 约定 | 目的 |
| --- | --- | --- |
| Python 入站请求 | 读取或生成 `X-Trace-Id` | 为一次用户请求确定追踪编号 |
| Python 日志 | 日志格式输出 `trace_id` | 让 Python Agent、工具调用、异常都能按 trace_id 搜索 |
| Python 调 Java | 请求头携带 `X-Trace-Id` | 把同一次链路传给 Java |
| Java Filter | 读取或生成 trace_id，写入 MDC | 让 Java 日志能输出同一个 trace_id |
| Java 响应头 | 返回 `X-Trace-Id` | 让 Python client、前端或网关能直接读取 |
| Java 响应体 | 返回 `trace_id` | 让业务错误响应能带排查编号 |
| Python Java client 日志 | 记录 `upstream_trace_id` | 确认 Java 返回的 trace_id 是否和 Python 当前链路一致 |

原则：

```text
trace_id 只用于排查，不用于认证或权限判断。
trace_id 不应该包含手机号、邮箱、token、订单明细等敏感信息。
成功响应和失败响应都应该带 trace_id。
缺少 trace_id 的请求仍应生成一个服务端 trace_id 用于排查，但 internal API 可以继续拒绝这类不合规调用。
```

## 7. 契约测试清单

后续实现阶段至少测试：

```text
订单查询成功响应结构符合契约。
订单不存在返回 404 + ORDER_NOT_FOUND。
无权限订单返回 403 + ORDER_ACCESS_DENIED。
订单号非法返回 422 + ORDER_ID_INVALID。
创建工单成功返回 201 + OK + ticket_id。
创建工单缺少 Idempotency-Key 返回 IDEMPOTENCY_KEY_REQUIRED。
同一幂等键同一参数重复请求返回同一 ticket_id。
同一幂等键不同参数返回 IDEMPOTENCY_KEY_CONFLICT。
Python client 能解析成功响应。
Python client 能把 Java 错误码映射为稳定 AppException。
```

阶段 7 第 11 节已新增共享契约测试入口：

| 入口 | 位置 | 作用 |
| --- | --- | --- |
| 共享契约用例 | `contracts/java-business-service/internal-api-contract-cases.json` | 把订单查询、无权限订单、创建工单、缺少幂等键等关键场景写成 Python 和 Java 都能读取的契约事实 |
| Java provider 契约测试 | `projects/java-business-service/src/test/java/com/panpan/aibusinessservice/InternalApiContractTest.java` | 验证真实 Java business service 返回的状态码、响应头、错误码和 data 字段符合契约 |
| Python consumer 契约模型 | `projects/ai-service/app/services/java_business_contract.py` | 用 Pydantic 校验 Java 成功响应 envelope 和 AI 工具可见字段 |
| Python consumer 契约测试 | `projects/ai-service/tests/test_java_business_contract.py` | 验证 Python 能接受合法 Java 响应，也能把 Java 错误码映射为稳定 `AppException` |

注意：当前 Python 运行时链路里仍保留历史 `java-mock-service` 调用代码。本节先把真实 Java business service 的契约锁住；后续迁移 Python 真实运行链路时，应以这份共享契约为准，逐步替换旧 mock 链路。

## 8. 阶段 7 完成状态

阶段 7 已完成：

```text
1. 真实 Spring Boot 业务服务骨架。
2. 传统 controller/service/mapper/entity/dto/config/exception/common 结构。
3. MyBatis Mapper + XML。
4. MySQL orders/tickets/ticket_events 业务数据表。
5. 订单查询读接口。
6. 创建工单写接口。
7. Redis 订单缓存、工单幂等缓存和工具限流。
8. internal token、allowed caller、user_id、tenant_id、trace_id header 边界。
9. Java 机器错误码和 Python 安全错误映射。
10. trace_id 跨 Python + Java 串联。
11. 共享契约 JSON、Java provider 契约测试、Python consumer 契约测试。
```

仍需注意：

```text
Python Agent 运行时主链路还没有完全从历史 java-mock-service 切换到 java-business-service。
真实用户表、完整登录认证、RBAC/ABAC 权限体系和前端客服工作台仍未完成。
```

## 9. 后续迁移顺序

后续如果要把 Python 运行时主链路从历史 `java-mock-service` 迁移到 `java-business-service`，建议顺序：

```text
1. 先确认共享契约文件覆盖必要成功和失败场景。
2. 让 Python Java client 默认调用 /internal/orders/{order_id} 和 /internal/tickets。
3. 保留 fake/mock 测试模式，避免自动化测试依赖真实 MySQL/Redis。
4. 调整 Agent 工具节点，使工具结果使用 Java business contract 的字段。
5. 调整错误处理，使 Java code 全部通过 java_error_mapping.py 映射。
6. 调整本地运行说明，把 Java mock 演示和真实 Java business 演示分清。
7. 调整 Agent eval 样本，避免旧 mock 数据和真实 Java 数据冲突。
8. 最后再清理或降级历史 java-mock-service 的角色。
```
