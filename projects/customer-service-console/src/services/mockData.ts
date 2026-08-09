export interface DashboardMetric {
  label: string
  value: string
  trend: string
  tone: 'blue' | 'green' | 'amber' | 'red'
}

export interface OrderRow {
  orderNo: string
  customer: string
  status: string
  amount: string
  logistics: string
}

export interface TicketRow {
  ticketNo: string
  title: string
  status: string
  priority: string
  owner: string
  updatedAt: string
}

export interface KnowledgeDocument {
  name: string
  domain: string
  status: string
  chunks: number
  updatedAt: string
}

export const dashboardMetrics: DashboardMetric[] = [
  { label: '今日 AI 会话', value: '128', trend: '+18%', tone: 'blue' },
  { label: '待处理工单', value: '24', trend: '-6', tone: 'amber' },
  { label: 'RAG 引用通过率', value: '97.8%', trend: '+1.2%', tone: 'green' },
  { label: 'LLM 失败率', value: '1.6%', trend: '-0.4%', tone: 'red' },
]

export const orderRows: OrderRow[] = [
  { orderNo: 'A1001', customer: 'U1001', status: '运输中', amount: '￥369.00', logistics: '杭州转运中心' },
  { orderNo: 'A1002', customer: 'U1001', status: '已签收', amount: '￥129.00', logistics: '上海浦东站点' },
  { orderNo: 'A1003', customer: 'U1002', status: '售后中', amount: '￥899.00', logistics: '待退货入仓' },
]

export const ticketRows: TicketRow[] = [
  {
    ticketNo: 'T20260804001',
    title: 'A1001 物流长时间未更新',
    status: 'OPEN',
    priority: 'normal',
    owner: '未分配',
    updatedAt: '2026-08-04 10:20',
  },
  {
    ticketNo: 'T20260803007',
    title: '退款到账时间咨询',
    status: 'IN_PROGRESS',
    priority: 'low',
    owner: '客服小李',
    updatedAt: '2026-08-04 09:12',
  },
  {
    ticketNo: 'T20260802011',
    title: '账号安全验证失败',
    status: 'WAITING_USER',
    priority: 'high',
    owner: '客服小王',
    updatedAt: '2026-08-03 18:42',
  },
]

export const knowledgeDocuments: KnowledgeDocument[] = [
  { name: '退款退货规则', domain: 'refund', status: '已入库', chunks: 5, updatedAt: '2026-08-03' },
  { name: '物流异常处理 SOP', domain: 'logistics', status: '已入库', chunks: 8, updatedAt: '2026-08-03' },
  { name: '账号安全常见问题', domain: 'security', status: '待重新入库', chunks: 4, updatedAt: '2026-08-01' },
]
