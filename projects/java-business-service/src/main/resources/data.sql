INSERT INTO app_roles (role_code, role_name, created_at)
SELECT 'customer', 'Customer', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (SELECT 1 FROM app_roles WHERE role_code = 'customer');

INSERT INTO app_roles (role_code, role_name, created_at)
SELECT 'agent', 'Customer Service Agent', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (SELECT 1 FROM app_roles WHERE role_code = 'agent');

INSERT INTO app_roles (role_code, role_name, created_at)
SELECT 'supervisor', 'Customer Service Supervisor', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (SELECT 1 FROM app_roles WHERE role_code = 'supervisor');

INSERT INTO app_roles (role_code, role_name, created_at)
SELECT 'admin', 'System Admin', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (SELECT 1 FROM app_roles WHERE role_code = 'admin');

INSERT INTO app_users (
  tenant_id,
  user_id,
  username,
  display_name,
  password_hash,
  status,
  created_at,
  updated_at
)
SELECT
  'default',
  'U1001',
  'customer',
  'Demo Customer',
  '{plain}123456',
  'active',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_users WHERE tenant_id = 'default' AND username = 'customer'
);

INSERT INTO app_users (
  tenant_id,
  user_id,
  username,
  display_name,
  password_hash,
  status,
  created_at,
  updated_at
)
SELECT
  'default',
  'U2001',
  'customer2',
  'Demo Customer 2',
  '{plain}123456',
  'active',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_users WHERE tenant_id = 'default' AND username = 'customer2'
);

INSERT INTO app_users (
  tenant_id,
  user_id,
  username,
  display_name,
  password_hash,
  status,
  created_at,
  updated_at
)
SELECT
  'default',
  'A1001',
  'agent',
  'Demo Agent',
  '{plain}123456',
  'active',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_users WHERE tenant_id = 'default' AND username = 'agent'
);

INSERT INTO app_users (
  tenant_id,
  user_id,
  username,
  display_name,
  password_hash,
  status,
  created_at,
  updated_at
)
SELECT
  'default',
  'S1001',
  'supervisor',
  'Demo Supervisor',
  '{plain}123456',
  'active',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_users WHERE tenant_id = 'default' AND username = 'supervisor'
);

INSERT INTO app_users (
  tenant_id,
  user_id,
  username,
  display_name,
  password_hash,
  status,
  created_at,
  updated_at
)
SELECT
  'default',
  'ADMIN',
  'admin',
  'Demo Admin',
  '{plain}123456',
  'active',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_users WHERE tenant_id = 'default' AND username = 'admin'
);

INSERT INTO app_user_roles (tenant_id, user_id, role_code, created_at)
SELECT 'default', 'U1001', 'customer', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_user_roles WHERE tenant_id = 'default' AND user_id = 'U1001' AND role_code = 'customer'
);

INSERT INTO app_user_roles (tenant_id, user_id, role_code, created_at)
SELECT 'default', 'U2001', 'customer', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_user_roles WHERE tenant_id = 'default' AND user_id = 'U2001' AND role_code = 'customer'
);

INSERT INTO app_user_roles (tenant_id, user_id, role_code, created_at)
SELECT 'default', 'A1001', 'agent', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_user_roles WHERE tenant_id = 'default' AND user_id = 'A1001' AND role_code = 'agent'
);

INSERT INTO app_user_roles (tenant_id, user_id, role_code, created_at)
SELECT 'default', 'S1001', 'supervisor', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_user_roles WHERE tenant_id = 'default' AND user_id = 'S1001' AND role_code = 'supervisor'
);

INSERT INTO app_user_roles (tenant_id, user_id, role_code, created_at)
SELECT 'default', 'ADMIN', 'admin', CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM app_user_roles WHERE tenant_id = 'default' AND user_id = 'ADMIN' AND role_code = 'admin'
);

INSERT INTO orders (
  tenant_id,
  order_id,
  user_id,
  order_status,
  payment_status,
  logistics_message,
  latest_event,
  can_create_ticket,
  amount,
  created_at,
  updated_at
)
SELECT
  'default',
  'A1001',
  'U1001',
  'shipped',
  'paid',
  '订单已发货，预计 2 天内送达。',
  '包裹已离开发货仓。',
  1,
  299.00,
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM orders WHERE tenant_id = 'default' AND order_id = 'A1001'
);

INSERT INTO orders (
  tenant_id,
  order_id,
  user_id,
  order_status,
  payment_status,
  logistics_message,
  latest_event,
  can_create_ticket,
  amount,
  created_at,
  updated_at
)
SELECT
  'default',
  'A1002',
  'U1001',
  'waiting_shipment',
  'paid',
  '商家已接单，等待仓库发货。',
  '仓库正在准备出库。',
  1,
  159.00,
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM orders WHERE tenant_id = 'default' AND order_id = 'A1002'
);

INSERT INTO orders (
  tenant_id,
  order_id,
  user_id,
  order_status,
  payment_status,
  logistics_message,
  latest_event,
  can_create_ticket,
  amount,
  created_at,
  updated_at
)
SELECT
  'default',
  'A2001',
  'U2001',
  'delivered',
  'paid',
  '订单已签收。',
  '用户已确认收货。',
  0,
  89.00,
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM orders WHERE tenant_id = 'default' AND order_id = 'A2001'
);

INSERT INTO knowledge_documents (
  tenant_id,
  document_id,
  title,
  doc_type,
  business_domain,
  permission_group,
  status,
  source_file_name,
  chunk_count,
  updated_by,
  created_at,
  updated_at
)
SELECT
  'default',
  'DOC_REFUND_POLICY',
  'Refund and Return Policy',
  'policy',
  'refund',
  'customer_service',
  'published',
  'refund-return-policy.md',
  5,
  'ADMIN',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM knowledge_documents WHERE tenant_id = 'default' AND document_id = 'DOC_REFUND_POLICY'
);

INSERT INTO knowledge_documents (
  tenant_id,
  document_id,
  title,
  doc_type,
  business_domain,
  permission_group,
  status,
  source_file_name,
  chunk_count,
  updated_by,
  created_at,
  updated_at
)
SELECT
  'default',
  'DOC_ACCOUNT_SECURITY',
  'Account Security FAQ',
  'faq',
  'account',
  'customer_service',
  'published',
  'account-security-faq.md',
  4,
  'ADMIN',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM knowledge_documents WHERE tenant_id = 'default' AND document_id = 'DOC_ACCOUNT_SECURITY'
);

INSERT INTO knowledge_documents (
  tenant_id,
  document_id,
  title,
  doc_type,
  business_domain,
  permission_group,
  status,
  source_file_name,
  chunk_count,
  updated_by,
  created_at,
  updated_at
)
SELECT
  'default',
  'DOC_LOGISTICS_POLICY',
  'Logistics Service Policy',
  'policy',
  'logistics',
  'customer_service',
  'draft',
  'logistics-service-policy.md',
  0,
  'S1001',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM knowledge_documents WHERE tenant_id = 'default' AND document_id = 'DOC_LOGISTICS_POLICY'
);

INSERT INTO tickets (
  tenant_id,
  ticket_id,
  requester_user_id,
  related_order_id,
  title,
  description,
  category,
  priority,
  ticket_status,
  source,
  confirmation_id,
  idempotency_key,
  request_fingerprint,
  created_trace_id,
  created_at,
  updated_at
)
SELECT
  'default',
  'T-DEMO-1001',
  'U1001',
  'A1001',
  'A1001 logistics has not updated',
  'Customer reports that order A1001 logistics has not updated for a long time.',
  'logistics',
  'normal',
  'created',
  'ai_agent',
  'demo-confirmation-1001',
  'demo-ticket-seed-1001',
  'demo-fingerprint-1001',
  'demo-trace-1001',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM tickets WHERE tenant_id = 'default' AND ticket_id = 'T-DEMO-1001'
);

INSERT INTO tickets (
  tenant_id,
  ticket_id,
  requester_user_id,
  related_order_id,
  title,
  description,
  category,
  priority,
  ticket_status,
  source,
  confirmation_id,
  idempotency_key,
  request_fingerprint,
  created_trace_id,
  created_at,
  updated_at
)
SELECT
  'default',
  'T-DEMO-2001',
  'U2001',
  'A2001',
  'Delivered order after-sales question',
  'Customer asks whether delivered order A2001 still supports after-sales service.',
  'after_sales',
  'low',
  'created',
  'manual',
  'demo-confirmation-2001',
  'demo-ticket-seed-2001',
  'demo-fingerprint-2001',
  'demo-trace-2001',
  CURRENT_TIMESTAMP(6),
  CURRENT_TIMESTAMP(6)
WHERE NOT EXISTS (
  SELECT 1 FROM tickets WHERE tenant_id = 'default' AND ticket_id = 'T-DEMO-2001'
);
