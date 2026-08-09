-- 已有库手动执行；新建库由 schema.sql 自动生效
ALTER TABLE orders ADD COLUMN amount DECIMAL(10,2) NOT NULL DEFAULT 0;
ALTER TABLE orders ADD COLUMN refund_amount DECIMAL(10,2) NULL;
ALTER TABLE orders ADD COLUMN refunded_at DATETIME(6) NULL;
ALTER TABLE orders ADD COLUMN refund_reason VARCHAR(255) NULL;

-- 退款审计事件表（Task 2 新增，生产库需补建；新建库由 schema.sql 自动生效）
CREATE TABLE IF NOT EXISTS order_events (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  event_id VARCHAR(64) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  event_payload JSON NOT NULL,
  operator_type VARCHAR(32) NOT NULL,
  operator_id VARCHAR(64) NOT NULL,
  trace_id VARCHAR(128) NOT NULL,
  idempotency_key VARCHAR(128) NULL,
  request_fingerprint VARCHAR(64) NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_order_events_tenant_event (tenant_id, event_id),
  UNIQUE KEY uk_order_events_tenant_idempotency (tenant_id, idempotency_key),
  KEY idx_order_events_tenant_order_created (tenant_id, order_id, created_at)
);

-- 订单取消状态列（Task 1 新增，生产库需补建；新建库由 schema.sql 自动生效）
ALTER TABLE orders ADD COLUMN canceled_at DATETIME(6) NULL;
ALTER TABLE orders ADD COLUMN cancel_reason VARCHAR(255) NULL;
