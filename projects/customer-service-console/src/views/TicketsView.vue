<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { addTicketMessage, getTicketDetail, listTickets, reopenTicket } from '../services/businessApi'
import type { TicketDetail, TicketEventItem, TicketListItem, TicketMessageItem } from '../services/businessApi'
import { useSessionStore } from '../stores/session'

const tickets = ref<TicketListItem[]>([])
const loading = ref(false)
const detailLoading = ref(false)
const ticketDetail = ref<TicketDetail | null>(null)
const detailVisible = ref(false)
const replySubmitting = ref(false)
const replyContent = ref('')
const session = useSessionStore()

type TimelineItem =
  | { kind: 'event'; createdAt: string; key: string; event: TicketEventItem }
  | { kind: 'message'; createdAt: string; key: string; message: TicketMessageItem }

const statusLabels: Record<string, string> = {
  created: '已创建',
  open: '待处理',
  in_progress: '处理中',
  waiting_user: '待用户补充',
  resolved: '已解决',
}

const priorityLabels: Record<string, string> = {
  low: '低',
  normal: '普通',
  high: '高',
}

async function loadTickets() {
  loading.value = true
  try {
    tickets.value = await listTickets()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '工单加载失败')
  } finally {
    loading.value = false
  }
}

function formatDate(value: string) {
  return value ? value.replace('T', ' ').slice(0, 19) : '-'
}

const timelineItems = computed<TimelineItem[]>(() => {
  if (!ticketDetail.value) {
    return []
  }
  return [
    ...ticketDetail.value.events.map((event) => ({
      kind: 'event' as const,
      createdAt: event.created_at,
      key: `event-${event.event_id}`,
      event,
    })),
    ...(ticketDetail.value.messages ?? []).map((message) => ({
      kind: 'message' as const,
      createdAt: message.created_at,
      key: `message-${message.message_id}`,
      message,
    })),
  ].sort((left, right) => left.createdAt.localeCompare(right.createdAt))
})

const canCustomerReply = computed(() => {
  const status = ticketDetail.value?.ticket_status
  return session.currentUser?.role === 'customer' && Boolean(status && ['created', 'in_progress', 'waiting_user'].includes(status))
})

const canCustomerReopen = computed(() => {
  return session.currentUser?.role === 'customer' && ticketDetail.value?.ticket_status === 'resolved'
})

const customerActionTitle = computed(() => (canCustomerReopen.value ? '申请重开工单' : '补充信息'))

const customerActionButtonText = computed(() => (canCustomerReopen.value ? '提交重开申请' : '提交补充信息'))

const customerReplyHint = computed(() => {
  if (ticketDetail.value?.ticket_status === 'waiting_user') {
    return '提交补充信息后，工单会自动恢复为“处理中”。'
  }
  if (canCustomerReply.value) {
    return '补充信息会立即展示给处理该工单的客服人员。'
  }
  if (canCustomerReopen.value) {
    return '如果问题仍未解决，请说明原因。提交后工单会重新进入“处理中”。'
  }
  return '已解决或已关闭的工单不能继续补充信息。'
})

function eventSummary(event: TicketEventItem) {
  if (event.event_type === 'ticket_resolved') {
    return '客服已发送解决说明，工单已标记为已解决'
  }
  if (event.event_type === 'ticket_reopened') {
    return '已申请重开工单，客服将继续处理'
  }
  if (!event.event_payload) {
    return event.event_type
  }
  try {
    const payload = JSON.parse(event.event_payload) as Record<string, unknown>
    if (typeof payload.target_status === 'string') {
      return `工单状态更新为：${statusLabels[payload.target_status] || payload.target_status}`
    }
    if (typeof payload.assignee_display_name === 'string') {
      return `已分配给：${payload.assignee_display_name}`
    }
    if (typeof payload.title === 'string') {
      return payload.title
    }
  } catch {
    return event.event_type
  }
  return event.event_type
}

async function openTicketDetail(ticket: TicketListItem) {
  detailVisible.value = true
  detailLoading.value = true
  replyContent.value = ''
  try {
    ticketDetail.value = await getTicketDetail(ticket.ticket_id)
  } catch (error) {
    detailVisible.value = false
    ElMessage.error(error instanceof Error ? error.message : '工单详情加载失败')
  } finally {
    detailLoading.value = false
  }
}

function messageTitle(message: TicketMessageItem) {
  return message.author_type === 'customer' ? '我的补充' : '客服回复'
}

async function submitCustomerAction() {
  if (!ticketDetail.value) {
    return
  }
  const content = replyContent.value.trim()
  if (!content) {
    ElMessage.warning('请输入补充信息')
    return
  }

  replySubmitting.value = true
  try {
    const reopening = canCustomerReopen.value
    ticketDetail.value = reopening
      ? await reopenTicket(ticketDetail.value.ticket_id, { content })
      : await addTicketMessage(ticketDetail.value.ticket_id, {
          visibility: 'public',
          content,
        })
    replyContent.value = ''
    await loadTickets()
    ElMessage.success(reopening ? '重开申请已提交' : '补充信息已提交')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '工单操作提交失败')
  } finally {
    replySubmitting.value = false
  }
}

onMounted(loadTickets)
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>我的工单</span>
        <el-button type="primary" plain :loading="loading" @click="loadTickets">刷新工单</el-button>
      </div>
    </template>

    <el-table v-loading="loading" :data="tickets" stripe @row-click="openTicketDetail">
      <el-table-column prop="ticket_id" label="工单号" width="160" />
      <el-table-column prop="title" label="标题" min-width="240" show-overflow-tooltip />
      <el-table-column prop="related_order_id" label="关联订单" width="140" />
      <el-table-column label="状态" width="120">
        <template #default="{ row }">
          <el-tag effect="light">{{ statusLabels[row.ticket_status] || row.ticket_status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="优先级" width="100">
        <template #default="{ row }">{{ priorityLabels[row.priority] || row.priority }}</template>
      </el-table-column>
      <el-table-column prop="requester_user_id" label="用户" width="120" />
      <el-table-column label="更新时间" width="180">
        <template #default="{ row }">{{ formatDate(row.updated_at) }}</template>
      </el-table-column>
    </el-table>

    <el-empty v-if="!loading && tickets.length === 0" description="暂无可见工单" />
  </el-card>

  <el-drawer v-model="detailVisible" title="工单详情" size="min(560px, 100%)" destroy-on-close>
    <div v-loading="detailLoading" class="ticket-detail">
      <template v-if="ticketDetail">
        <div class="detail-title">
          <h2>{{ ticketDetail.title }}</h2>
          <el-tag>{{ statusLabels[ticketDetail.ticket_status] || ticketDetail.ticket_status }}</el-tag>
        </div>
        <p class="ticket-description">{{ ticketDetail.description }}</p>

        <section v-if="session.currentUser?.role === 'customer'" class="detail-section">
          <h3>{{ customerActionTitle }}</h3>
          <el-alert :title="customerReplyHint" type="info" :closable="false" show-icon />
          <div class="reply-editor">
            <el-input
              v-model="replyContent"
              type="textarea"
              :rows="4"
              maxlength="2000"
              show-word-limit
              :disabled="!canCustomerReply && !canCustomerReopen"
              :placeholder="canCustomerReopen ? '说明问题仍未解决的原因，以及希望客服继续处理的内容' : '补充订单信息、问题细节或客服需要的材料说明'"
            />
            <div class="reply-actions">
              <el-button type="primary" :disabled="!canCustomerReply && !canCustomerReopen" :loading="replySubmitting" @click="submitCustomerAction">
                {{ customerActionButtonText }}
              </el-button>
            </div>
          </div>
        </section>

        <section class="detail-section">
          <h3>处理进展</h3>
          <el-timeline>
            <el-timeline-item
              v-for="item in timelineItems"
              :key="item.key"
              :timestamp="formatDate(item.createdAt)"
            >
              <template v-if="item.kind === 'event'">
                <strong>{{ eventSummary(item.event) }}</strong>
              </template>
              <template v-else>
                <strong>{{ messageTitle(item.message) }}</strong>
                <p class="message-content">{{ item.message.content }}</p>
                <small>{{ item.message.author_display_name }}</small>
              </template>
            </el-timeline-item>
          </el-timeline>
          <el-empty v-if="timelineItems.length === 0" description="暂无处理记录" />
        </section>
      </template>
    </div>
  </el-drawer>
</template>

<style scoped>
.ticket-detail {
  min-height: 280px;
}

.detail-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.detail-title h2 {
  margin: 0;
  font-size: 18px;
  line-height: 1.45;
}

.ticket-description,
.message-content {
  white-space: pre-wrap;
  line-height: 1.7;
}

.detail-section {
  border-top: 1px solid #e5e7eb;
  margin-top: 20px;
  padding-top: 16px;
}

.detail-section h3 {
  margin: 0 0 12px;
  font-size: 15px;
}

.reply-editor {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.reply-actions {
  display: flex;
  justify-content: flex-end;
}

small {
  color: #6b7280;
}
</style>
