CREATE TABLE IF NOT EXISTS app_users (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  username VARCHAR(64) NOT NULL,
  display_name VARCHAR(100) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  status VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_app_users_tenant_user (tenant_id, user_id),
  UNIQUE KEY uk_app_users_tenant_username (tenant_id, username),
  KEY idx_app_users_tenant_status (tenant_id, status)
);

CREATE TABLE IF NOT EXISTS app_roles (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  role_code VARCHAR(64) NOT NULL,
  role_name VARCHAR(100) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_app_roles_code (role_code)
);

CREATE TABLE IF NOT EXISTS app_user_roles (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  role_code VARCHAR(64) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_app_user_roles_tenant_user_role (tenant_id, user_id, role_code),
  KEY idx_app_user_roles_tenant_role (tenant_id, role_code)
);

CREATE TABLE IF NOT EXISTS orders (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  order_status VARCHAR(32) NOT NULL,
  payment_status VARCHAR(32) NOT NULL,
  logistics_message VARCHAR(255) NOT NULL,
  latest_event VARCHAR(255) NOT NULL,
  can_create_ticket TINYINT(1) NOT NULL DEFAULT 1,
  amount DECIMAL(10,2) NOT NULL DEFAULT 0,
  refund_amount DECIMAL(10,2) NULL,
  refunded_at DATETIME(6) NULL,
  refund_reason VARCHAR(255) NULL,
  canceled_at DATETIME(6) NULL,
  cancel_reason VARCHAR(255) NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_orders_tenant_order (tenant_id, order_id),
  KEY idx_orders_tenant_user_order (tenant_id, user_id, order_id)
);

CREATE TABLE IF NOT EXISTS tickets (
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

CREATE TABLE IF NOT EXISTS ticket_events (
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

CREATE TABLE IF NOT EXISTS ticket_assignments (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  ticket_id VARCHAR(64) NOT NULL,
  assignee_user_id VARCHAR(64) NOT NULL,
  assignee_display_name VARCHAR(100) NOT NULL,
  assigned_by_user_id VARCHAR(64) NOT NULL,
  assigned_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_ticket_assignments_tenant_ticket (tenant_id, ticket_id),
  KEY idx_ticket_assignments_tenant_assignee (tenant_id, assignee_user_id)
  );

  CREATE TABLE IF NOT EXISTS ticket_messages (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    tenant_id VARCHAR(64) NOT NULL,
    message_id VARCHAR(64) NOT NULL,
    ticket_id VARCHAR(64) NOT NULL,
    visibility VARCHAR(16) NOT NULL,
    content VARCHAR(2000) NOT NULL,
    author_type VARCHAR(32) NOT NULL,
    author_user_id VARCHAR(64) NOT NULL,
    author_display_name VARCHAR(100) NOT NULL,
    trace_id VARCHAR(128) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    UNIQUE KEY uk_ticket_messages_tenant_message (tenant_id, message_id),
    KEY idx_ticket_messages_tenant_ticket_created (tenant_id, ticket_id, created_at)
  );

  CREATE TABLE IF NOT EXISTS knowledge_documents (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  document_id VARCHAR(64) NOT NULL,
  title VARCHAR(200) NOT NULL,
  doc_type VARCHAR(32) NOT NULL,
  business_domain VARCHAR(64) NOT NULL,
  permission_group VARCHAR(64) NOT NULL,
  status VARCHAR(32) NOT NULL,
  source_file_name VARCHAR(255) NOT NULL,
  chunk_count INT NOT NULL DEFAULT 0,
  updated_by VARCHAR(64) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_knowledge_documents_tenant_doc (tenant_id, document_id),
  KEY idx_knowledge_documents_tenant_status (tenant_id, status),
  KEY idx_knowledge_documents_tenant_permission (tenant_id, permission_group)
);

CREATE TABLE IF NOT EXISTS ai_conversations (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  conversation_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  title VARCHAR(200) NOT NULL,
  conversation_status VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_ai_conversations_tenant_conversation (tenant_id, conversation_id),
  KEY idx_ai_conversations_tenant_user_updated (tenant_id, user_id, updated_at)
);

CREATE TABLE IF NOT EXISTS ai_messages (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  message_id VARCHAR(64) NOT NULL,
  conversation_id VARCHAR(64) NOT NULL,
  sender_type VARCHAR(32) NOT NULL,
  content TEXT NOT NULL,
  trace_id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_ai_messages_tenant_message (tenant_id, message_id),
  KEY idx_ai_messages_tenant_conversation_created (tenant_id, conversation_id, created_at)
);

CREATE TABLE IF NOT EXISTS ai_response_feedback (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id VARCHAR(64) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  conversation_id VARCHAR(128) NOT NULL,
  trace_id VARCHAR(128) NOT NULL,
  rating VARCHAR(16) NOT NULL,
  reason VARCHAR(64) NULL,
  agent_route VARCHAR(64) NOT NULL,
  citation_count INT NOT NULL DEFAULT 0,
  human_handoff_suggested TINYINT(1) NOT NULL DEFAULT 0,
  user_message_excerpt VARCHAR(1000) NULL,
  assistant_answer_excerpt VARCHAR(2000) NULL,
  citation_summary_json TEXT NULL,
  review_status VARCHAR(32) NOT NULL DEFAULT 'candidate',
  bad_case_id VARCHAR(160) NULL,
  reviewed_by_user_id VARCHAR(64) NULL,
  reviewed_at DATETIME(6) NULL,
  review_note VARCHAR(1000) NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  UNIQUE KEY uk_ai_response_feedback_identity (tenant_id, user_id, conversation_id, trace_id),
  KEY idx_ai_response_feedback_tenant_created (tenant_id, created_at),
  KEY idx_ai_response_feedback_tenant_rating (tenant_id, rating, created_at)
);

CREATE TABLE IF NOT EXISTS rag_eval_runs (
  run_id VARCHAR(64) PRIMARY KEY,
  retriever VARCHAR(16) NOT NULL COMMENT 'keyword|vector',
  started_at DATETIME(3) NOT NULL,
  completed_at DATETIME(3) NOT NULL,
  top_k INT NOT NULL,
  hit_rate DOUBLE NOT NULL,
  recall DOUBLE NOT NULL,
  `precision` DOUBLE NOT NULL COMMENT 'precision 是 MySQL 保留字，需反引号',
  mrr DOUBLE NOT NULL,
  case_count INT NOT NULL,
  details_json TEXT NOT NULL COMMENT '每个 case 的检索明细（JSON）',
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  KEY idx_rag_eval_runs_retriever_started (retriever, started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='RAG 检索评测结果（双模式）';

CREATE TABLE IF NOT EXISTS ai_cost_records (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  model VARCHAR(64) NOT NULL,
  intent VARCHAR(64) NOT NULL,
  call_count INT NOT NULL DEFAULT 0,
  input_tokens BIGINT NOT NULL DEFAULT 0,
  output_tokens BIGINT NOT NULL DEFAULT 0,
  total_tokens BIGINT NOT NULL DEFAULT 0,
  estimated_cost DECIMAL(12, 4) NOT NULL DEFAULT 0,
  window_start DATETIME(3) NOT NULL,
  window_end DATETIME(3) NOT NULL,
  UNIQUE KEY uk_model_intent_window (model, intent, window_start),
  KEY idx_ai_cost_records_model (model),
  KEY idx_ai_cost_records_intent (intent)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI LLM 调用成本聚合（模型 × 意图）';

CREATE TABLE IF NOT EXISTS ai_human_handoffs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  conversation_id VARCHAR(128) NOT NULL,
  user_id VARCHAR(64) NOT NULL,
  tenant_id VARCHAR(64) NOT NULL,
  reason VARCHAR(300) NOT NULL,
  related_order_id VARCHAR(64) NULL,
  emotion VARCHAR(32) NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'pending',
  assigned_agent VARCHAR(64) NULL,
  note VARCHAR(500) NULL,
  created_at DATETIME(3) NOT NULL,
  updated_at DATETIME(3) NOT NULL,
  resolved_at DATETIME(3) NULL,
  KEY idx_ai_human_handoffs_status_created (status, created_at),
  KEY idx_ai_human_handoffs_conversation (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI 转人工坐席队列';
