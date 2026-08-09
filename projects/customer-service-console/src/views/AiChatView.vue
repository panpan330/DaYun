<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Close, Loading, Plus, Select } from '@element-plus/icons-vue'
import {
  correctConsoleAgentTicketConfirmation,
  decideConsoleAgentTicketConfirmation,
  getConsoleAgentConversation,
  listConsoleAgentConversations,
  requestConsoleAgentHumanHandoff,
  streamConsoleAgentMessage,
  submitConsoleAgentFeedback,
} from '../services/aiChatApi'
import type {
  ConsoleAgentResponse,
  ConsoleAgentConversationMessage,
  ConsoleAgentConversationSummary,
  ConsoleAgentFeedbackReason,
  ConsoleAgentHumanHandoff,
  ConsoleAgentTicketConfirmation,
  ConsoleAgentTicketFields,
  ConsoleAgentStreamStage,
  CreatedTicket,
  RagCitation,
} from '../services/aiChatApi'
import { useSessionStore } from '../stores/session'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'human_agent'
  text: string
  traceId?: string
  citations?: RagCitation[]
  suggestions?: string[]
  pendingConfirmation?: ConsoleAgentTicketConfirmation
  createdTicket?: CreatedTicket
  humanHandoff?: ConsoleAgentHumanHandoff
  emotion?: string
  feedbackRating?: 'helpful' | 'unhelpful'
  feedbackReason?: ConsoleAgentFeedbackReason
}

const input = ref('A1001 的物流一直没有更新，我想投诉并让客服处理。')
const sessionStore = useSessionStore()
const loading = ref(false)
const agentActivity = ref<ConsoleAgentStreamStage>()
const confirmationSubmitting = ref(false)
const editingConfirmationMessageId = ref<string>()
const correctionFields = ref<ConsoleAgentTicketFields>()
const conversationId = ref<string>()
const lastTraceId = ref<string>()
const conversationSummaries = ref<ConsoleAgentConversationSummary[]>([])
const conversationLoading = ref(false)
const feedbackSubmittingMessageId = ref<string>()
const feedbackReasonMessageId = ref<string>()

const messages = ref<ChatMessage[]>([
  {
    id: 'welcome',
    role: 'assistant',
    text: '你好，我是 AI 客服助手。我会根据问题自动查询订单、检索知识库，或在你确认后创建工单。',
  },
])

onMounted(() => {
  void restoreConversation()
})

async function sendMessage() {
  const message = input.value.trim()
  if (!message || loading.value || confirmationSubmitting.value) {
    return
  }

  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    text: message,
  })
  input.value = ''
  loading.value = true
  agentActivity.value = { stage: 'preparing', label: '正在准备本次请求' }

  try {
    appendAgentResponse(
      await streamConsoleAgentMessage(
        {
          message,
          conversationId: conversationId.value,
        },
        (stage) => {
          agentActivity.value = stage
        },
      ),
    )
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'AI 客服请求失败')
  } finally {
    loading.value = false
    agentActivity.value = undefined
  }
}

async function decideConfirmation(message: ChatMessage, approved: boolean) {
  const confirmation = message.pendingConfirmation
  if (!confirmation || !conversationId.value || confirmationSubmitting.value) {
    return
  }

  confirmationSubmitting.value = true
  try {
    const response = await decideConsoleAgentTicketConfirmation({
      conversationId: conversationId.value,
      confirmationId: confirmation.confirmation_id,
      approved,
    })
    message.pendingConfirmation = undefined
    const kind = confirmationKind(confirmation)
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      text: approved
        ? (kind === 'cancel' ? '确认取消订单' : kind === 'refund' ? '确认退款' : '确认创建工单')
        : (kind === 'cancel' ? '取消取消' : kind === 'refund' ? '取消退款' : '取消创建工单'),
    })
    appendAgentResponse(response)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '工单确认操作失败')
  } finally {
    confirmationSubmitting.value = false
  }
}

async function requestHumanHandoff(message: ChatMessage) {
  if (!message.humanHandoff || !conversationId.value || confirmationSubmitting.value) {
    return
  }

  confirmationSubmitting.value = true
  try {
    const response = await requestConsoleAgentHumanHandoff(conversationId.value)
    message.humanHandoff = undefined
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      text: '请求转人工客服处理',
    })
    appendAgentResponse(response)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '无法发起人工转接')
  } finally {
    confirmationSubmitting.value = false
  }
}

function beginConfirmationCorrection(message: ChatMessage) {
  const confirmation = message.pendingConfirmation
  if (!confirmation || confirmationSubmitting.value) {
    return
  }
  editingConfirmationMessageId.value = message.id
  correctionFields.value = { ...confirmation.ticket_fields }
}

const emotionLabels: Record<string, string> = {
  angry: '😠 愤怒',
  anxious: '😰 焦虑',
  dissatisfied: '😒 不满',
  urgent: '⚡ 急切',
  apologetic: '🙏 歉意',
  neutral: '😐 中性',
  satisfied: '😊 满意',
}

function emotionLabel(emotion: string): string {
  return emotionLabels[emotion] || emotion
}

function emotionTagType(emotion: string): 'danger' | 'warning' | 'info' | 'success' {
  if (emotion === 'angry' || emotion === 'anxious' || emotion === 'urgent') return 'danger'
  if (emotion === 'dissatisfied') return 'warning'
  if (emotion === 'satisfied') return 'success'
  return 'info'
}

function cancelConfirmationCorrection() {
  editingConfirmationMessageId.value = undefined
  correctionFields.value = undefined
}

async function submitConfirmationCorrection(message: ChatMessage) {
  const confirmation = message.pendingConfirmation
  const ticketFields = correctionFields.value
  if (!confirmation || !ticketFields || !conversationId.value || confirmationSubmitting.value) {
    return
  }

  confirmationSubmitting.value = true
  try {
    const response = await correctConsoleAgentTicketConfirmation({
      conversationId: conversationId.value,
      confirmationId: confirmation.confirmation_id,
      ticketFields,
    })
    message.pendingConfirmation = undefined
    const kind = confirmationKind(confirmation)
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      text:
        kind === 'cancel'
          ? '修改取消信息并重新确认'
          : kind === 'refund'
            ? '修改退款信息并重新确认'
            : '修改工单草稿并重新确认',
    })
    cancelConfirmationCorrection()
    appendAgentResponse(response)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '工单草稿修改失败')
  } finally {
    confirmationSubmitting.value = false
  }
}

function appendAgentResponse(response: ConsoleAgentResponse) {
  conversationId.value = response.conversation_id
  persistActiveConversationId(response.conversation_id)
  lastTraceId.value = response.trace_id
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'assistant',
    text: response.reply,
    traceId: response.trace_id,
    citations: response.citations,
    suggestions: response.suggestions,
    pendingConfirmation: response.pending_ticket_confirmation || undefined,
    createdTicket: response.created_ticket || undefined,
    humanHandoff: response.human_handoff || undefined,
    emotion: response.emotion || undefined,
  })
  void refreshConversationSummaries()
}

async function submitFeedback(
  message: ChatMessage,
  rating: 'helpful' | 'unhelpful',
  reason?: ConsoleAgentFeedbackReason,
) {
  if (!conversationId.value || !message.traceId || feedbackSubmittingMessageId.value) {
    return
  }
  feedbackSubmittingMessageId.value = message.id
  try {
    const response = await submitConsoleAgentFeedback({
      conversationId: conversationId.value,
      traceId: message.traceId,
      rating,
      reason,
    })
    message.feedbackRating = response.rating
    message.feedbackReason = response.reason || undefined
    feedbackReasonMessageId.value = undefined
    ElMessage.success('反馈已保存')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '反馈提交失败')
  } finally {
    feedbackSubmittingMessageId.value = undefined
  }
}

function openUnhelpfulFeedback(message: ChatMessage) {
  feedbackReasonMessageId.value = feedbackReasonMessageId.value === message.id ? undefined : message.id
}

async function restoreConversation() {
  await refreshConversationSummaries()
  const savedConversationId = localStorage.getItem(activeConversationStorageKey())
  const candidate = conversationSummaries.value.find(
    (conversation) => conversation.conversation_id === savedConversationId,
  ) || conversationSummaries.value[0]
  if (candidate) {
    await loadConversation(candidate.conversation_id)
  }
}

async function refreshConversationSummaries() {
  try {
    conversationSummaries.value = await listConsoleAgentConversations()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '无法读取最近会话')
  }
}

async function loadConversation(targetConversationId: string) {
  if (conversationLoading.value || targetConversationId === conversationId.value) {
    return
  }
  conversationLoading.value = true
  try {
    const conversation = await getConsoleAgentConversation(targetConversationId)
    conversationId.value = conversation.conversation_id
    persistActiveConversationId(conversation.conversation_id)
    messages.value = conversation.messages.map(toChatMessage)
    lastTraceId.value = [...conversation.messages]
      .reverse()
      .find((message) => message.trace_id)?.trace_id || undefined
  } catch (error) {
    localStorage.removeItem(activeConversationStorageKey())
    ElMessage.error(error instanceof Error ? error.message : '无法恢复会话记录')
  } finally {
    conversationLoading.value = false
  }
}

function startNewConversation() {
  if (loading.value || confirmationSubmitting.value) {
    return
  }
  conversationId.value = undefined
  lastTraceId.value = undefined
  localStorage.removeItem(activeConversationStorageKey())
  messages.value = [
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      text: '你好，我是 AI 客服助手。我会根据问题自动查询订单、检索知识库，或在你确认后创建工单。',
    },
  ]
}

function toChatMessage(message: ConsoleAgentConversationMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role,
    text: message.content,
    traceId: message.trace_id || undefined,
    citations: message.citations,
    suggestions: message.suggestions,
    pendingConfirmation: message.pending_ticket_confirmation || undefined,
    createdTicket: message.created_ticket || undefined,
    humanHandoff: message.human_handoff || undefined,
    emotion: message.emotion || undefined,
    feedbackRating: message.feedback_rating || undefined,
    feedbackReason: message.feedback_reason || undefined,
  }
}

function activeConversationStorageKey() {
  const user = sessionStore.currentUser
  return `customer-service-console-agent-conversation:${user?.tenantId || 'default'}:${user?.id || 'anonymous'}`
}

function persistActiveConversationId(value: string) {
  localStorage.setItem(activeConversationStorageKey(), value)
}

function formatConversationTime(value: string) {
  return new Date(value).toLocaleString('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function formatCitation(citation: RagCitation) {
  const title = citation.title || citation.source
  const section = citation.section ? ` / ${citation.section}` : ''
  return `${citation.source_index}. ${title}${section}`
}

function priorityLabel(priority: string) {
  return { low: '低', normal: '普通', high: '高' }[priority] || priority
}

function isRefundConfirmation(confirmation?: ConsoleAgentTicketConfirmation): boolean {
  // The backend's is_refund_execution flag is the reliable discriminator (the
  // refund *execution* flow vs a refund-typed ticket draft). Whenever the flag
  // is present — the backend always writes it explicitly — it wins; the draft
  // issue_type is only a fallback for legacy confirmations recorded before the
  // flag existed (where a refund-typed *ticket* would otherwise be mistaken
  // for a refund execution).
  if (confirmation?.is_refund_execution !== undefined) {
    return confirmation.is_refund_execution === true
  }
  return confirmation?.ticket_fields?.issue_type === 'refund'
}

function isCancelConfirmation(confirmation?: ConsoleAgentTicketConfirmation): boolean {
  // Symmetric to isRefundConfirmation: the backend's is_cancel_execution flag
  // wins whenever present; issue_type falls back to 'cancel' only for legacy
  // confirmations recorded before the flag existed.
  if (confirmation?.is_cancel_execution !== undefined) {
    return confirmation.is_cancel_execution === true
  }
  return confirmation?.ticket_fields?.issue_type === 'cancel'
}

type ConfirmationKind = 'cancel' | 'refund' | 'ticket'

function confirmationKind(confirmation?: ConsoleAgentTicketConfirmation): ConfirmationKind {
  if (isCancelConfirmation(confirmation)) return 'cancel'
  if (isRefundConfirmation(confirmation)) return 'refund'
  return 'ticket'
}

function confirmationTitle(confirmation?: ConsoleAgentTicketConfirmation): string {
  if (confirmationKind(confirmation) === 'cancel') return '确认取消订单'
  if (confirmationKind(confirmation) === 'refund') return '确认退款'
  return confirmation?.title || '确认创建工单'
}

function confirmationConfirmLabel(confirmation?: ConsoleAgentTicketConfirmation): string {
  if (confirmationKind(confirmation) === 'cancel') return '确认取消'
  if (confirmationKind(confirmation) === 'refund') return '确认退款'
  return '确认创建工单'
}
</script>

<template>
  <section class="chat-layout">
    <el-card class="chat-panel" shadow="never">
      <template #header>
        <div class="card-header">
          <span>AI 客服对话</span>
          <el-tag type="success">统一 Agent</el-tag>
        </div>
      </template>

      <div class="message-list">
        <div v-for="message in messages" :key="message.id" class="message-row" :class="message.role">
          <div class="message-bubble">
            <div
              v-if="message.role === 'human_agent' || (message.emotion && message.role === 'assistant')"
              class="message-meta"
            >
              <el-tag v-if="message.role === 'human_agent'" size="small" type="success" effect="light">
                人工客服
              </el-tag>
              <el-tag
                v-if="message.emotion && message.role === 'assistant'"
                size="small"
                :type="emotionTagType(message.emotion)"
                class="emotion-tag"
              >
                {{ emotionLabel(message.emotion) }}
              </el-tag>
            </div>
            <p class="message-text">{{ message.text }}</p>

            <div v-if="message.pendingConfirmation" class="agent-confirmation">
              <div class="confirmation-heading">
                <strong>{{ confirmationTitle(message.pendingConfirmation) }}</strong>
                <el-tag type="warning">等待确认</el-tag>
              </div>
              <p>{{ message.pendingConfirmation.summary }}</p>
              <el-form
                v-if="editingConfirmationMessageId === message.id && correctionFields"
                class="correction-form"
                label-position="top"
              >
                <template v-if="isCancelConfirmation(message.pendingConfirmation)">
                  <el-form-item label="关联订单">
                    <el-input v-model="correctionFields.order_id" maxlength="64" />
                  </el-form-item>
                  <el-form-item label="取消原因">
                    <el-input v-model="correctionFields.description" type="textarea" :rows="3" maxlength="200" show-word-limit />
                  </el-form-item>
                </template>
                <template v-else-if="isRefundConfirmation(message.pendingConfirmation)">
                  <el-form-item label="退款订单">
                    <el-input v-model="correctionFields.order_id" maxlength="64" />
                  </el-form-item>
                  <el-form-item label="退款原因">
                    <el-input v-model="correctionFields.description" type="textarea" :rows="3" maxlength="1000" show-word-limit />
                  </el-form-item>
                </template>
                <template v-else>
                  <el-form-item label="问题类型">
                    <el-select v-model="correctionFields.issue_type">
                      <el-option label="退款/退货" value="refund" />
                      <el-option label="物流/发货" value="logistics" />
                      <el-option label="投诉/异常处理" value="complaint" />
                      <el-option label="知识库缺口" value="policy_gap" />
                    </el-select>
                  </el-form-item>
                  <el-form-item label="关联订单">
                    <el-input v-model="correctionFields.order_id" maxlength="64" />
                  </el-form-item>
                  <el-form-item label="问题说明">
                    <el-input v-model="correctionFields.description" type="textarea" :rows="3" maxlength="1000" show-word-limit />
                  </el-form-item>
                  <el-form-item label="处理诉求">
                    <el-input v-model="correctionFields.user_request" maxlength="200" />
                  </el-form-item>
                  <el-form-item label="紧急程度">
                    <el-radio-group v-model="correctionFields.urgency">
                      <el-radio value="low">低</el-radio>
                      <el-radio value="normal">普通</el-radio>
                      <el-radio value="high">高</el-radio>
                    </el-radio-group>
                  </el-form-item>
                  <el-form-item label="需要人工复核">
                    <el-switch v-model="correctionFields.need_human_review" />
                  </el-form-item>
                </template>
              </el-form>

              <el-descriptions v-else :column="1" size="small" border>
                <template v-if="isCancelConfirmation(message.pendingConfirmation)">
                  <el-descriptions-item label="订单号">
                    {{ message.pendingConfirmation.ticket_fields.order_id || '未提供' }}
                  </el-descriptions-item>
                  <el-descriptions-item label="取消原因">
                    {{ message.pendingConfirmation.ticket_fields.description }}
                  </el-descriptions-item>
                </template>
                <template v-else-if="isRefundConfirmation(message.pendingConfirmation)">
                  <el-descriptions-item label="退款订单">
                    {{ message.pendingConfirmation.ticket_fields.order_id || '未提供' }}
                  </el-descriptions-item>
                  <el-descriptions-item label="退款原因">
                    {{ message.pendingConfirmation.ticket_fields.description }}
                  </el-descriptions-item>
                </template>
                <template v-else>
                  <el-descriptions-item label="问题类型">
                    {{ message.pendingConfirmation.ticket_fields.issue_type }}
                  </el-descriptions-item>
                  <el-descriptions-item label="关联订单">
                    {{ message.pendingConfirmation.ticket_fields.order_id || '未提供' }}
                  </el-descriptions-item>
                  <el-descriptions-item label="紧急程度">
                    {{ priorityLabel(message.pendingConfirmation.ticket_fields.urgency) }}
                  </el-descriptions-item>
                  <el-descriptions-item label="问题说明">
                    {{ message.pendingConfirmation.ticket_fields.description }}
                  </el-descriptions-item>
                </template>
              </el-descriptions>
              <div class="confirmation-actions">
                <template v-if="editingConfirmationMessageId === message.id">
                  <el-button :disabled="confirmationSubmitting" @click="cancelConfirmationCorrection">取消修改</el-button>
                  <el-button type="primary" :loading="confirmationSubmitting" @click="submitConfirmationCorrection(message)">
                    保存修改并重新确认
                  </el-button>
                </template>
                <template v-else>
                  <el-button :loading="confirmationSubmitting" @click="decideConfirmation(message, false)">取消</el-button>
                  <el-button :disabled="confirmationSubmitting" @click="beginConfirmationCorrection(message)">修改信息</el-button>
                  <el-button type="primary" :loading="confirmationSubmitting" @click="decideConfirmation(message, true)">
                    {{ confirmationConfirmLabel(message.pendingConfirmation) }}
                  </el-button>
                </template>
              </div>
            </div>

            <el-alert
              v-if="message.createdTicket"
              class="created-ticket"
              type="success"
              :closable="false"
              show-icon
              :title="`工单 ${message.createdTicket.ticket_id} 已创建`"
            >
              <template #default>
                <RouterLink to="/tickets">查看我的工单</RouterLink>
              </template>
            </el-alert>

            <el-alert
              v-if="message.humanHandoff"
              class="human-handoff"
              type="warning"
              :closable="false"
              show-icon
              title="建议转人工处理"
            >
              <template #default>
                <p>{{ message.humanHandoff.reason }}</p>
                <el-button
                  type="warning"
                  plain
                  :loading="confirmationSubmitting"
                  @click="requestHumanHandoff(message)"
                >
                  转人工并生成工单草稿
                </el-button>
              </template>
            </el-alert>

            <div v-if="message.citations?.length" class="citation-list">
              <strong>引用来源</strong>
              <span v-for="citation in message.citations" :key="citation.chunk_id">
                {{ formatCitation(citation) }}
              </span>
            </div>
            <div v-if="message.suggestions?.length" class="suggestion-list">
              <strong>建议</strong>
              <span v-for="suggestion in message.suggestions" :key="suggestion">{{ suggestion }}</span>
            </div>
            <div v-if="message.role === 'assistant' && message.traceId" class="feedback-area">
              <div class="feedback-actions">
                <el-tooltip content="这条回答有帮助">
                  <el-button
                    circle
                    plain
                    size="small"
                    :icon="Select"
                    :type="message.feedbackRating === 'helpful' ? 'success' : 'default'"
                    :loading="feedbackSubmittingMessageId === message.id"
                    :disabled="Boolean(feedbackSubmittingMessageId)"
                    @click="submitFeedback(message, 'helpful')"
                  />
                </el-tooltip>
                <el-tooltip content="这条回答没有帮助">
                  <el-button
                    circle
                    plain
                    size="small"
                    :icon="Close"
                    :type="message.feedbackRating === 'unhelpful' ? 'danger' : 'default'"
                    :disabled="Boolean(feedbackSubmittingMessageId)"
                    @click="openUnhelpfulFeedback(message)"
                  />
                </el-tooltip>
              </div>
              <div v-if="feedbackReasonMessageId === message.id" class="feedback-reason">
                <el-select v-model="message.feedbackReason" clearable placeholder="可选：反馈原因" size="small">
                  <el-option label="回答不准确" value="answer_incorrect" />
                  <el-option label="没有理解问题" value="intent_misunderstood" />
                  <el-option label="引用不相关" value="citation_irrelevant" />
                  <el-option label="应该转人工" value="should_handoff" />
                  <el-option label="工单流程不正确" value="ticket_flow_incorrect" />
                  <el-option label="其他" value="other" />
                </el-select>
                <el-button
                  size="small"
                  type="danger"
                  :loading="feedbackSubmittingMessageId === message.id"
                  @click="submitFeedback(message, 'unhelpful', message.feedbackReason)"
                >
                  提交
                </el-button>
              </div>
            </div>
            <span v-if="message.traceId" class="message-trace">trace_id: {{ message.traceId }}</span>
          </div>
        </div>
        <div v-if="loading && agentActivity" class="agent-activity">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>{{ agentActivity.label }}</span>
        </div>
      </div>

      <div class="chat-input">
        <el-input
          v-model="input"
          type="textarea"
          :rows="3"
          resize="none"
          :disabled="loading || confirmationSubmitting"
          @keydown.ctrl.enter.prevent="sendMessage"
        />
        <el-button type="primary" :loading="loading" :disabled="confirmationSubmitting" @click="sendMessage">
          发送
        </el-button>
      </div>
    </el-card>

    <aside class="agent-session-panel">
      <div class="conversation-panel-heading">
        <h2>本次会话</h2>
        <el-tooltip content="新建会话">
          <el-button :icon="Plus" circle plain title="新建会话" @click="startNewConversation" />
        </el-tooltip>
      </div>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="Agent API">POST /api/ai/agent/conversations/stream</el-descriptions-item>
        <el-descriptions-item label="会话 ID">{{ conversationId || '首次发送后生成' }}</el-descriptions-item>
        <el-descriptions-item label="最近 trace">{{ lastTraceId || '-' }}</el-descriptions-item>
      </el-descriptions>
      <el-divider content-position="left">最近会话</el-divider>
      <div class="conversation-list" v-loading="conversationLoading">
        <el-button
          v-for="conversation in conversationSummaries"
          :key="conversation.conversation_id"
          class="conversation-list-item"
          text
          :type="conversation.conversation_id === conversationId ? 'primary' : 'default'"
          @click="loadConversation(conversation.conversation_id)"
        >
          <span>{{ conversation.title }}</span>
          <small>{{ formatConversationTime(conversation.updated_at) }}</small>
        </el-button>
        <el-empty v-if="!conversationLoading && !conversationSummaries.length" :image-size="56" description="暂无历史会话" />
      </div>
    </aside>
  </section>
</template>

<style scoped>
.chat-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: 20px;
}

.chat-panel { min-height: 640px; }
.card-header, .confirmation-heading, .confirmation-actions, .chat-input { display: flex; align-items: center; }
.card-header, .confirmation-heading { justify-content: space-between; gap: 12px; }
.message-list { display: flex; min-height: 470px; flex-direction: column; gap: 14px; overflow-y: auto; }
.message-row { display: flex; }
.message-row.user { justify-content: flex-end; }
.message-bubble { max-width: min(760px, 88%); border: 1px solid var(--el-border-color); padding: 12px; background: var(--el-fill-color-light); }
.message-row.user .message-bubble { background: var(--el-color-primary-light-9); }
.message-row.human_agent .message-bubble {
  background: var(--el-color-success-light-9);
  border-color: var(--el-color-success-light-5);
}
.message-meta {
  margin-bottom: 4px;
}
.emotion-tag {
  margin-right: 8px;
}
.message-text { margin: 0; white-space: pre-wrap; line-height: 1.65; }
.agent-confirmation, .citation-list, .suggestion-list { display: grid; gap: 8px; margin-top: 12px; }
.agent-activity { display: inline-flex; align-items: center; gap: 8px; margin-top: 12px; color: var(--el-color-primary); font-size: 13px; }
.agent-confirmation { border-top: 1px solid var(--el-border-color); padding-top: 12px; }
.agent-confirmation p { margin: 0; color: var(--el-text-color-regular); }
.correction-form { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 12px; }
.correction-form :deep(.el-form-item:nth-child(3)), .correction-form :deep(.el-form-item:nth-child(4)) { grid-column: 1 / -1; }
.confirmation-actions { justify-content: flex-end; gap: 8px; }
.created-ticket { margin-top: 12px; }
.human-handoff { margin-top: 12px; }
.human-handoff :deep(.el-alert__content) { display: grid; gap: 10px; }
.human-handoff p { margin: 0; }
.citation-list, .suggestion-list { color: var(--el-text-color-regular); font-size: 13px; }
.feedback-area { display: grid; gap: 8px; margin-top: 12px; }
.feedback-actions, .feedback-reason { display: flex; align-items: center; gap: 8px; }
.feedback-reason :deep(.el-select) { width: min(240px, 100%); }
.message-trace { display: block; margin-top: 10px; color: var(--el-text-color-secondary); font-size: 12px; }
.chat-input { align-items: flex-end; gap: 12px; margin-top: 18px; }
.chat-input :deep(.el-textarea) { flex: 1; }
.agent-session-panel { border-left: 1px solid var(--el-border-color); padding-left: 20px; }
.conversation-panel-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.agent-session-panel h2 { margin: 0; font-size: 16px; }
.conversation-list { display: grid; gap: 4px; min-height: 64px; }
.conversation-list-item { display: grid; justify-content: stretch; gap: 4px; width: 100%; height: auto; padding: 8px; text-align: left; white-space: normal; }
.conversation-list-item span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.conversation-list-item small { color: var(--el-text-color-secondary); font-size: 11px; }

@media (max-width: 960px) {
  .chat-layout { grid-template-columns: 1fr; }
  .correction-form { grid-template-columns: 1fr; }
  .agent-session-panel { border-left: 0; border-top: 1px solid var(--el-border-color); padding: 16px 0 0; }
}
</style>
