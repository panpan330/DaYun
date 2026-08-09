<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { listKnowledgeDocuments, listOrders, listTickets } from '../services/businessApi'
import { getEvaluationOverview } from '../services/evaluationApi'
import { getOpsStatsSummary } from '../services/aiChatApi'
import type { OpsStatsSummary } from '../services/aiChatApi'
import { aiApi } from '../services/http'
import type { OrderListItem, TicketListItem } from '../services/businessApi'
import type { EvaluationOverview } from '../services/evaluationApi'

interface DependencyStatus {
  name: string
  status: 'ok' | 'unreachable' | 'not_configured'
  message: string
  latency_ms: number | null
}

interface DependencyOverview {
  status: 'ok' | 'degraded'
  service: string
  dependencies: DependencyStatus[]
  time: string
}

const dependencyLabels: Record<string, string> = {
  java_business: 'Java 业务服务',
  mcp_product: 'MCP 工具服务',
  redis: 'Redis',
  qdrant: 'Qdrant',
}

function dependencyTagType(status: string): 'success' | 'danger' | 'info' {
  if (status === 'ok') return 'success'
  if (status === 'unreachable') return 'danger'
  return 'info'
}

const dependencies = ref<DependencyStatus[]>([])
const dependencyOverall = ref<'ok' | 'degraded'>('ok')
const dependencyError = ref(false)

async function loadDependencies() {
  try {
    const response = await aiApi.get<DependencyOverview>('/api/ai/health/dependencies')
    dependencyOverall.value = response.data.status
    dependencies.value = response.data.dependencies
    dependencyError.value = false
  } catch {
    dependencyError.value = true
  }
}

function refreshAll() {
  void loadDashboard()
  void loadDependencies()
  void loadCostOverview()
  void loadOpsStats()
}

const orders = ref<OrderListItem[]>([])
const tickets = ref<TicketListItem[]>([])
const knowledgeDocumentCount = ref(0)
const evaluationOverview = ref<EvaluationOverview | null>(null)
const loading = ref(false)

interface CostSummaryRow {
  model?: string
  intent?: string
  callCount: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  estimatedCost: number
}

interface CostOverview {
  by_model: CostSummaryRow[]
  by_intent: CostSummaryRow[]
  totals: CostSummaryRow
}

const costOverview = ref<CostOverview | null>(null)
const costError = ref(false)

const opsDays = ref(7)
const opsStats = ref<OpsStatsSummary | null>(null)
const opsLoading = ref(false)
const opsError = ref(false)

const emotionLabels: Record<string, string> = {
  angry: '愤怒',
  anxious: '焦虑',
  dissatisfied: '不满',
  urgent: '急切',
  apologetic: '歉意',
  neutral: '中性',
  satisfied: '满意',
  unknown: '未知',
}

async function loadOpsStats() {
  opsLoading.value = true
  opsError.value = false
  try {
    opsStats.value = await getOpsStatsSummary(opsDays.value)
  } catch {
    opsError.value = true
  } finally {
    opsLoading.value = false
  }
}

function helpfulRatePercent(): number {
  const rate = opsStats.value?.feedback.helpful_rate
  return rate == null ? 0 : Math.round(rate * 100)
}

watch(opsDays, () => {
  void loadOpsStats()
})

async function loadCostOverview() {
  try {
    const response = await aiApi.get<CostOverview>('/api/ai/cost/overview')
    costOverview.value = response.data
    costError.value = false
  } catch {
    costError.value = true
  }
}

const openTicketCount = computed(() => {
  return tickets.value.filter((ticket) =>
    ['created', 'in_progress', 'waiting_user'].includes(ticket.ticket_status),
  ).length
})

const recentTickets = computed(() => tickets.value.slice(0, 6))

const metrics = computed(() => [
  {
    label: '可见订单',
    value: String(orders.value.length),
    tag: '真实 Java',
    type: 'success' as const,
  },
  {
    label: '待处理工单',
    value: String(openTicketCount.value),
    tag: `${tickets.value.length} total`,
    type: openTicketCount.value > 0 ? ('warning' as const) : ('success' as const),
  },
  {
    label: '知识库文档',
    value: String(knowledgeDocumentCount.value),
    tag: 'Java + AI',
    type: 'info' as const,
  },
  {
    label: '评估通过率',
    value:
      evaluationOverview.value?.latest_run.metrics.find((metric) => metric.name === 'check_pass_rate')
        ?.display_value || '-',
    tag: evaluationOverview.value?.latest_run.passed ? '通过' : '有失败',
    type: evaluationOverview.value?.latest_run.passed ? ('success' as const) : ('danger' as const),
  },
])

const statusLabels: Record<string, string> = {
  created: '待处理',
  in_progress: '处理中',
  waiting_user: '待用户补充',
  resolved: '已解决',
  closed: '已关闭',
}

const priorityLabels: Record<string, string> = {
  low: '低',
  normal: '普通',
  high: '高',
}

async function loadDashboard() {
  loading.value = true
  try {
    const [orderList, ticketList, knowledgeDocuments, evaluation] = await Promise.all([
      listOrders(),
      listTickets(),
      listKnowledgeDocuments(),
      getEvaluationOverview(),
    ])
    orders.value = orderList
    tickets.value = ticketList
    knowledgeDocumentCount.value = knowledgeDocuments.length
    evaluationOverview.value = evaluation
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '运营概览加载失败')
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  void loadDashboard()
  void loadDependencies()
  void loadCostOverview()
  void loadOpsStats()
})
</script>

<template>
  <section v-loading="loading" class="dashboard-page">
    <section class="page-grid">
      <el-card v-for="metric in metrics" :key="metric.label" class="metric-card" shadow="never">
        <p>{{ metric.label }}</p>
        <div>
          <strong>{{ metric.value }}</strong>
          <el-tag size="small" :type="metric.type">{{ metric.tag }}</el-tag>
        </div>
      </el-card>
    </section>

    <section class="content-grid two-columns">
      <el-card shadow="never">
        <template #header>
          <div class="card-header">
            <span>项目运行链路</span>
            <el-button type="primary" plain :loading="loading" @click="refreshAll">刷新</el-button>
          </div>
        </template>
        <el-timeline v-if="!dependencyError">
          <el-timeline-item
            v-for="dep in dependencies"
            :key="dep.name"
            :timestamp="dependencyLabels[dep.name] || dep.name"
            :type="dep.status === 'ok' ? 'success' : (dep.status === 'unreachable' ? 'danger' : 'info')"
          >
            <div class="dependency-row">
              <el-tag :type="dependencyTagType(dep.status)" size="small" effect="light">
                {{ dep.status === 'ok' ? '正常' : (dep.status === 'unreachable' ? '不可达' : '未配置') }}
              </el-tag>
              <span v-if="dep.latency_ms !== null" class="dependency-latency">{{ dep.latency_ms }}ms</span>
              <el-tooltip :content="dep.message" placement="top">
                <span class="dependency-message">{{ dep.message }}</span>
              </el-tooltip>
            </div>
          </el-timeline-item>
          <el-timeline-item :timestamp="'整体状态'">
            <el-tag :type="dependencyOverall === 'ok' ? 'success' : 'danger'" effect="dark">
              {{ dependencyOverall === 'ok' ? '运行正常' : '依赖异常' }}
            </el-tag>
          </el-timeline-item>
        </el-timeline>
        <el-timeline v-else>
          <el-timeline-item timestamp="前端控制台" type="primary">
            Vue3 页面已接入 Java public API 和 Python AI API。
          </el-timeline-item>
          <el-timeline-item timestamp="Java 业务服务" type="success">
            登录、订单、工单、知识库元数据、工单状态流转均来自 Spring Boot + MyBatis。
          </el-timeline-item>
          <el-timeline-item timestamp="Python AI 服务" type="warning">
            AI 对话、RAG 问答、知识库入库、评估与 bad case 看板均由 FastAPI 提供。
          </el-timeline-item>
          <el-timeline-item timestamp="真实依赖" type="info">
            MySQL 保存业务数据，Redis 支撑缓存/幂等/限流，Qdrant 支撑真实 RAG 检索。
          </el-timeline-item>
        </el-timeline>
      </el-card>

      <el-card shadow="never">
        <template #header>最近工单</template>
        <el-table :data="recentTickets" size="small" height="300">
          <el-table-column prop="ticket_id" label="工单号" width="150" />
          <el-table-column prop="title" label="标题" min-width="180" show-overflow-tooltip />
          <el-table-column label="状态" width="110">
            <template #default="{ row }">
              <el-tag effect="light">{{ statusLabels[row.ticket_status] || row.ticket_status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="优先级" width="90">
            <template #default="{ row }">{{ priorityLabels[row.priority] || row.priority }}</template>
          </el-table-column>
        </el-table>
        <el-empty v-if="!loading && recentTickets.length === 0" description="暂无可见工单" />
      </el-card>
    </section>

    <el-card shadow="never">
      <template #header>当前评估快照</template>
      <el-descriptions v-if="evaluationOverview" :column="4" border>
        <el-descriptions-item label="run_id">{{ evaluationOverview.latest_run.run_id }}</el-descriptions-item>
        <el-descriptions-item label="数据集">
          {{ evaluationOverview.latest_run.dataset_name }}:{{ evaluationOverview.latest_run.dataset_version }}
        </el-descriptions-item>
        <el-descriptions-item label="检查项">{{ evaluationOverview.latest_run.evaluated_check_count }}</el-descriptions-item>
        <el-descriptions-item label="失败项">{{ evaluationOverview.latest_run.failed_check_count }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card shadow="never">
      <template #header>成本概览（估算）</template>
      <el-empty v-if="costError" description="成本数据暂不可用" />
      <template v-else-if="costOverview">
        <el-table :data="costOverview.by_model" size="small" max-height="200">
          <el-table-column prop="model" label="模型" min-width="120" />
          <el-table-column prop="callCount" label="调用次数" width="90" />
          <el-table-column prop="totalTokens" label="总 Token" width="110" />
          <el-table-column label="估算费用" width="120">
            <template #default="{ row }">{{ row.estimatedCost?.toFixed(4) }}</template>
          </el-table-column>
        </el-table>
        <el-table :data="costOverview.by_intent" size="small" max-height="200" style="margin-top: 12px">
          <el-table-column prop="intent" label="意图" min-width="120" />
          <el-table-column prop="callCount" label="调用次数" width="90" />
          <el-table-column prop="totalTokens" label="总 Token" width="110" />
          <el-table-column label="估算费用" width="120">
            <template #default="{ row }">{{ row.estimatedCost?.toFixed(4) }}</template>
          </el-table-column>
        </el-table>
        <el-descriptions v-if="costOverview.totals" :column="3" border style="margin-top: 12px">
          <el-descriptions-item label="总调用">{{ costOverview.totals.callCount }}</el-descriptions-item>
          <el-descriptions-item label="总 Token">{{ costOverview.totals.totalTokens }}</el-descriptions-item>
          <el-descriptions-item label="估算费用">${{ costOverview.totals.estimatedCost?.toFixed(4) }}</el-descriptions-item>
        </el-descriptions>
      </template>
    </el-card>

    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>运营概览</span>
          <el-segmented
            v-model="opsDays"
            :options="[
              { label: '近 7 天', value: 7 },
              { label: '近 30 天', value: 30 },
            ]"
          />
        </div>
      </template>
      <div v-loading="opsLoading">
        <el-empty v-if="opsError" description="运营数据暂不可用" />
        <template v-else-if="opsStats">
          <div class="ops-section">
            <div class="ops-title">满意度</div>
            <el-progress
              :percentage="helpfulRatePercent()"
              :status="helpfulRatePercent() >= 60 ? 'success' : 'warning'"
            />
            <div class="ops-sub">
              helpful {{ opsStats.feedback.helpful }} / unhelpful {{ opsStats.feedback.unhelpful }}
            </div>
          </div>
          <div class="ops-section">
            <div class="ops-title">转人工（共 {{ opsStats.handoffs.total }}）</div>
            <el-tag type="warning" effect="light" class="emotion-tag">待处理 {{ opsStats.handoffs.pending }}</el-tag>
            <el-tag type="primary" effect="light" class="emotion-tag">处理中 {{ opsStats.handoffs.in_progress }}</el-tag>
            <el-tag type="success" effect="light" class="emotion-tag">已关闭 {{ opsStats.handoffs.closed }}</el-tag>
          </div>
          <div class="ops-section">
            <div class="ops-title">情绪分布</div>
            <el-tag
              v-for="(count, emotion) in opsStats.emotion_distribution"
              :key="emotion"
              effect="plain"
              class="emotion-tag"
            >
              {{ emotionLabels[emotion] || emotion }} {{ count }}
            </el-tag>
          </div>
          <div class="ops-section">
            <div class="ops-title">每日对话量</div>
            <div v-for="(count, day) in opsStats.conversation_volume" :key="day" class="volume-row">
              <span>{{ day }}</span>
              <span>{{ count }} 条</span>
            </div>
          </div>
        </template>
      </div>
    </el-card>
  </section>
</template>

<style scoped>
.dashboard-page {
  display: grid;
  gap: 16px;
}
.ops-section {
  margin-bottom: 16px;
}
.ops-title {
  font-weight: 600;
  margin-bottom: 8px;
}
.ops-sub {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  margin-top: 4px;
}
.emotion-tag {
  margin-right: 6px;
  margin-bottom: 4px;
}
.volume-row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  padding: 2px 0;
}
.dependency-row {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dependency-latency {
  color: #909399;
  font-size: 12px;
}
.dependency-message {
  color: #909399;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 320px;
}
</style>
