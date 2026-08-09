# Java Business Service 数据库设计草案

本文档是阶段 7 第 4 节的配套设计文档。

它的作用不是替代学习笔记，而是把 `projects/java-business-service` 后续接入 MySQL 时要使用的核心表、字段、索引、唯一约束和边界规则固定下来。

当前项目仍处在学习和作品化阶段，所以本文档采用“够真实、够清晰、后续能落地”的设计方式，不追求一开始就覆盖复杂电商系统里的所有表。

当前落地状态：

```text
阶段 7 第 5 节已落地 orders 表。
GET /internal/orders/{order_id} 已通过 OrderMapper + OrderMapper.xml 从 MySQL 读取。
阶段 7 第 6 节已落地 tickets 表和 ticket_events 表。
POST /internal/tickets 已通过 TicketMapper + TicketMapper.xml 写入 MySQL。
阶段 7 第 7 节已接入 Redis 订单缓存、工单幂等缓存和工具接口限流。
阶段 7 第 7.5 节已完成 Java 服务结构传统化重构，并用 MyBatis 替换 JdbcTemplate。
users 仍是设计草案，后续阶段继续落地。
```

---

## 一、设计目标

阶段 7 的目标是把原来的 Java mock service 逐步升级成真实 Java Spring Boot 业务服务。

在 AI Agent 场景里，数据库设计不能只考虑传统 CRUD，还要额外考虑：

```text
模型不能直接信任。
模型不能直接写库。
Python AI 服务不能绕过 Java 业务边界。
Java 后端要能校验权限、幂等、业务规则和审计字段。
MySQL 要保存长期业务事实。
Redis 负责短期状态、缓存、限流和幂等加速。
```

因此本阶段先设计四张核心表：

| 表 | 作用 |
| --- | --- |
| `users` | 保存用户身份、租户归属和基础状态 |
| `orders` | 保存订单基础状态，用于订单查询读工具 |
| `tickets` | 保存客服工单当前状态，用于创建工单写工具 |
| `ticket_events` | 保存工单生命周期事件，用于审计、追踪和排查 |

---

## 二、核心边界

### 1. MySQL 不直接服务模型

MySQL 服务的是 Java 业务系统。

模型想查订单或创建工单，真实链路应该是：

```text
用户问题
-> Python AI 服务判断意图
-> Tool Calling 生成工具请求
-> Python 后端校验工具名和参数
-> Python 调用 Java internal API
-> Java 校验 internal token、tenant、user、业务规则
-> Java 查询或写入 MySQL
-> Java 返回字段白名单 DTO
-> Python 再交给模型组织中文回答
```

关键点：

```text
模型不拿数据库账号。
模型不拼 SQL。
模型不看 Entity 全字段。
模型只接触工具结果里允许暴露的字段。
```

### 2. AI 写操作必须可审计

创建工单属于写操作。

写操作必须能回答这些问题：

```text
是谁触发的？
属于哪个租户？
对应哪个用户？
是不是 AI Agent 创建的？
用户是否确认过？
重复提交时如何处理？
这次请求的 trace_id 是什么？
工单创建后发生过哪些状态变化？
```

所以 `tickets` 表需要：

```text
tenant_id
requester_user_id
source
confirmation_id
idempotency_key
created_trace_id
```

`ticket_events` 表需要：

```text
event_type
event_payload
operator_type
operator_id
trace_id
created_at
```

### 3. 租户和用户边界必须进入索引

本项目即使只是学习系统，也要从一开始保留真实业务边界：

```text
tenant_id 决定数据属于哪个租户。
user_id 决定数据属于哪个用户。
order_id / ticket_id 是对外业务 ID。
数据库自增 id 只是内部主键，不作为 AI 工具契约字段。
```

所以常见查询索引不能只建在 `order_id` 或 `ticket_id` 上，而要带上 `tenant_id`，必要时还要带上 `user_id`。

---

## 三、表关系概览

```text
users
  tenant_id + user_id 唯一

orders
  tenant_id + order_id 唯一
  tenant_id + user_id 用于权限归属判断

tickets
  tenant_id + ticket_id 唯一
  tenant_id + idempotency_key 唯一
  related_order_id 可以关联 orders.order_id
  requester_user_id 可以关联 users.user_id

ticket_events
  tenant_id + event_id 唯一
  tenant_id + ticket_id 关联 tickets.ticket_id
```

当前设计先不使用数据库物理外键。

原因：

```text
学习阶段先把查询、写入、幂等、权限和 AI 调用链路跑通。
Java 层会做逻辑校验。
后续如果需要更强数据库约束，可以再补外键或迁移脚本。
```

这不是说外键不好，而是本项目当前重点是 AI Agent 调用真实 Java 后端的工程边界。

---

## 四、DDL 草案

### 1. users

```sql
CREATE TABLE users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  display_name VARCHAR(100) NOT NULL,
  role VARCHAR(32) NOT NULL,
  status VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_users_tenant_user (tenant_id, user_id),
  KEY idx_users_tenant_status (tenant_id, status)
);
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `id` | 数据库内部主键 |
| `tenant_id` | 租户 ID，用于隔离不同租户数据 |
| `user_id` | 对外用户 ID，Python AI 服务传递这个字段 |
| `display_name` | 用户展示名 |
| `role` | 用户角色，例如 customer、staff、admin |
| `status` | 用户状态，例如 active、disabled |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |

约束说明：

```text
unique(tenant_id, user_id)
```

表示同一个租户下用户 ID 不能重复。

不要只用 `user_id` 做全局唯一，是为了给多租户系统留下空间。

---

### 2. orders

```sql
CREATE TABLE orders (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  order_status VARCHAR(32) NOT NULL,
  payment_status VARCHAR(32) NOT NULL,
  logistics_message VARCHAR(255) NOT NULL,
  latest_event VARCHAR(255) NOT NULL,
  can_create_ticket TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_orders_tenant_order (tenant_id, order_id),
  KEY idx_orders_tenant_user_order (tenant_id, user_id, order_id)
);
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `id` | 数据库内部主键 |
| `tenant_id` | 租户 ID |
| `order_id` | 对外订单 ID |
| `user_id` | 订单所属用户 ID |
| `order_status` | 订单状态 |
| `payment_status` | 支付状态 |
| `logistics_message` | 物流或履约信息 |
| `latest_event` | 最近业务事件摘要 |
| `can_create_ticket` | 当前订单是否允许创建工单 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |

查询模式：

```text
AI Agent 查询订单时，Java 后端应该按 tenant_id + user_id + order_id 查询。
```

这样可以避免用户 A 通过猜测订单号查到用户 B 的订单。

---

### 3. tickets

```sql
CREATE TABLE tickets (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  ticket_id VARCHAR(64) NOT NULL,
  requester_user_id VARCHAR(64) NOT NULL,
  related_order_id VARCHAR(64) NULL,
  title VARCHAR(200) NOT NULL,
  description VARCHAR(1000) NOT NULL,
  category VARCHAR(32) NOT NULL,
  priority VARCHAR(32) NOT NULL,
  ticket_status VARCHAR(32) NOT NULL,
  source VARCHAR(32) NOT NULL,
  confirmation_id VARCHAR(64) NOT NULL,
  idempotency_key VARCHAR(128) NOT NULL,
  request_fingerprint VARCHAR(64) NOT NULL,
  created_trace_id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_tickets_tenant_ticket (tenant_id, ticket_id),
  UNIQUE KEY uk_tickets_tenant_idempotency (tenant_id, idempotency_key),
  KEY idx_tickets_tenant_requester_created (tenant_id, requester_user_id, created_at),
  KEY idx_tickets_tenant_order_category (tenant_id, related_order_id, category)
);
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `id` | 数据库内部主键 |
| `tenant_id` | 租户 ID |
| `ticket_id` | 对外工单 ID |
| `requester_user_id` | 发起工单的用户 |
| `related_order_id` | 关联订单 ID，可以为空 |
| `title` | 工单标题 |
| `description` | 工单描述 |
| `category` | 工单分类 |
| `priority` | 优先级 |
| `ticket_status` | 工单当前状态 |
| `source` | 创建来源，例如 ai_agent、manual、system |
| `confirmation_id` | 用户确认凭证 |
| `idempotency_key` | 幂等键 |
| `request_fingerprint` | 幂等请求指纹，用于判断同一个幂等键是否对应同一个请求 |
| `created_trace_id` | 创建请求对应的 trace_id |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |

关键约束：

```text
unique(tenant_id, idempotency_key)
```

它是创建工单防重复的数据库兜底。

即使 Redis 后续做了幂等缓存，也不能只靠 Redis。Redis 可能过期、宕机、丢失或出现并发穿透，最终仍要让 MySQL 的唯一约束保证同一租户下同一个幂等键不会创建多个工单。

---

### 4. ticket_events

```sql
CREATE TABLE ticket_events (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  event_id VARCHAR(64) NOT NULL,
  ticket_id VARCHAR(64) NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  event_payload JSON NOT NULL,
  operator_type VARCHAR(32) NOT NULL,
  operator_id VARCHAR(64) NOT NULL,
  trace_id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_ticket_events_tenant_event (tenant_id, event_id),
  KEY idx_ticket_events_tenant_ticket_created (tenant_id, ticket_id, created_at)
);
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `id` | 数据库内部主键 |
| `tenant_id` | 租户 ID |
| `event_id` | 对外事件 ID |
| `ticket_id` | 对应工单 ID |
| `event_type` | 事件类型，例如 created、assigned、status_changed、closed |
| `event_payload` | 事件详情 JSON |
| `operator_type` | 操作者类型，例如 ai_agent、user、staff、system |
| `operator_id` | 操作者 ID |
| `trace_id` | 触发本事件的请求链路 ID |
| `created_at` | 事件发生时间 |

为什么需要事件表：

```text
tickets 保存当前状态。
ticket_events 保存历史过程。
```

如果只保存 `tickets`，后续很难回答：

```text
工单是谁创建的？
是不是 AI 创建的？
用户什么时候确认的？
创建后有没有人工客服处理过？
某个 trace_id 对应哪些业务变化？
```

---

## 五、索引设计原则

### 1. 索引跟查询场景走

当前最核心的查询场景：

| 查询场景 | 推荐索引 |
| --- | --- |
| 根据用户身份查询订单 | `orders(tenant_id, user_id, order_id)` |
| 根据订单业务 ID 定位订单 | `orders(tenant_id, order_id)` 唯一索引 |
| 根据工单业务 ID 定位工单 | `tickets(tenant_id, ticket_id)` 唯一索引 |
| 创建工单幂等判断 | `tickets(tenant_id, idempotency_key)` 唯一索引 |
| 查询用户最近工单 | `tickets(tenant_id, requester_user_id, created_at)` |
| 查询工单事件时间线 | `ticket_events(tenant_id, ticket_id, created_at)` |

### 2. 不给所有字段都建索引

这些字段暂时不建普通索引：

```text
description
logistics_message
latest_event
event_payload
```

原因：

```text
它们是长文本或 JSON。
普通 B-Tree 索引不一定适合。
过多索引会增加写入成本。
AI 场景下真正高频查询通常围绕 tenant_id、user_id、order_id、ticket_id、created_at 和 idempotency_key。
```

### 3. 唯一索引是业务规则的一部分

唯一索引不是单纯为了查询快。

在本设计里，它还表达业务规则：

| 唯一索引 | 业务含义 |
| --- | --- |
| `uk_users_tenant_user` | 同租户用户唯一 |
| `uk_orders_tenant_order` | 同租户订单唯一 |
| `uk_tickets_tenant_ticket` | 同租户工单唯一 |
| `uk_tickets_tenant_idempotency` | 同租户同一写请求不能重复创建工单 |
| `uk_ticket_events_tenant_event` | 同租户事件 ID 唯一 |

---

## 六、和 Java 代码的映射

当前 Java 内存模型已经有这些领域对象：

| Java 领域对象 | 后续 MySQL 表 |
| --- | --- |
| `Order` | `orders` |
| `Ticket` | `tickets` |
| 后续新增 `User` | `users` |
| 后续新增 `TicketEvent` | `ticket_events` |

当前 Repository 落地状态：

```text
OrderMapper 默认通过 OrderMapper.xml 查询 orders。
TicketMapper 默认通过 TicketMapper.xml 查询和写入 tickets、ticket_events。
旧 Repository / JdbcTemplate / memory 实现已经在第 7.5 节移除。
```

第 5 节已把订单查询读工具真实化：

```text
OrderMapper 接口保持订单查询能力
OrderMapper.xml 承载订单查询 SQL
GET /internal/orders/{order_id} 仍然保持契约不变
Controller 不应该因为换数据库而大幅变化
```

第 6 节已把创建工单写工具真实化：

```text
TicketMapper 承载工单写入和事件写入能力
TicketMapper.xml 承载 tickets 和 ticket_events SQL
创建工单写入 tickets 表
同时写入 ticket_events 表
使用 request_fingerprint 判断幂等键是否对应同一个请求
使用 MySQL 唯一索引作为幂等最终兜底
```

这体现一个重要工程原则：

```text
对 Python AI 服务来说，Java API 契约要稳定；
对 Java 内部来说，数据来源可以从内存切换到 MySQL。
```

---

## 七、Redis key 契约

Redis 不是 MySQL 表，但它也是系统契约的一部分。

当前阶段使用这些 key：

| 场景 | Redis key | 默认 TTL | 作用 |
| --- | --- | --- | --- |
| 订单查询缓存 | `java-business:order:{tenant_id}:{order_id}` | 300 秒 | 减少重复订单查询打到 MySQL |
| 创建工单幂等缓存 | `java-business:ticket-idempotency:{tenant_id}:{idempotency_key}` | 86400 秒 | 保存 `request_fingerprint` 和 `ticket_id`，加速重复写请求判断 |
| internal 工具限流 | `java-business:rate-limit:{tenant_id}:{user_id}:{method}:{uri}` | 60 秒 | 使用 Redis `INCR` + TTL 做 fixed window 计数 |

关键边界：

```text
订单缓存只缓存订单数据，不缓存权限结果。
幂等缓存只做加速，最终仍以 MySQL unique(tenant_id, idempotency_key) 兜底。
限流 key 按 tenant、user、method、uri 拆开，避免不同用户或不同工具互相挤占额度。
Redis key 里的动态片段统一做 URL encode，避免 URI 等特殊字符破坏 key 结构。
Redis 故障时，缓存和幂等缓存降级回 MySQL；限流当前选择 warning 后放行。
```

这体现一个重要工程原则：

```text
MySQL 保存长期业务事实。
Redis 保存短期加速和保护状态。
```

---

## 八、后续接入顺序

阶段 7 后续建议按这个顺序落地：

1. 第 5 节：接入 MySQL，先让订单查询从 MySQL 读取。已完成。
2. 第 6 节：让创建工单写入 MySQL，并写入 `ticket_events`。已完成。
3. 第 7 节：接入 Redis，做幂等加速、查询缓存和限流。已完成。
4. 第 7.5 节：Java 服务结构传统化重构 + MyBatis。已完成。
5. 第 8 节：完善 internal token、用户身份、租户和权限边界。
6. 第 9 节：把 Java 错误码映射成 AI 用户可理解的回答。
7. 第 10 节：让 trace_id 串联 Python、Java、MySQL 和 Redis。
8. 第 11 节：补契约测试和集成测试。
9. 第 12 节：阶段 7 项目整理。

---

## 九、当前版本的取舍

本设计暂时不做：

```text
完整商品表
完整支付表
完整物流轨迹表
完整客服人员排班表
完整权限角色系统
数据库物理外键
分库分表
读写分离
Flyway/Liquibase 迁移体系
```

这些都不是不重要，而是当前阶段的核心不是搭一个完整电商数据库，而是学习：

```text
AI Agent 如何通过稳定、安全、可追踪的 Java API 操作真实业务数据。
```

等订单查询、创建工单、Redis 幂等和 trace 链路都跑通后，再扩展更多业务表会更自然。
