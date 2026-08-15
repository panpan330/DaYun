import { aiApi } from './http'
import type { ApiResponse } from './http'

export type ChatRole = 'user' | 'assistant' | 'human_agent'

export interface ChatHistoryMessage {
  role: ChatRole
  content: string
}

export interface ConsoleChatResponse {
  reply: string
  conversation_id: string
  trace_id: string
  mode: string
}

export interface ConsoleAgentTicketFields {
  issue_type: string
  order_id: string | null
  description: string
  user_request: string
  urgency: string
  need_human_review: boolean
}

export interface ConsoleAgentTicketConfirmation {
  confirmation_id: string
  title: string
  summary: string
  /**
   * True only for a refund *execution* confirmation (backend graph state
   * `refund_request_active`).  Refund-typed tickets created through the
   * ordinary ticket flow leave this absent/false.  The browser prefers this
   * flag over `ticket_fields.issue_type` to pick refund-vs-ticket copy.
   */
  is_refund_execution?: boolean
  /**
   * True only for a cancel *execution* confirmation (backend graph state
   * `cancel_request_active`).  Cancel-typed tickets created through the
   * ordinary ticket flow leave this absent/false.  The browser prefers this
   * flag over `ticket_fields.issue_type` to pick cancel-vs-ticket copy.
   */
  is_cancel_execution?: boolean
  ticket_fields: ConsoleAgentTicketFields
}

export interface ConsoleAgentHumanHandoff {
  reason: string
  related_order_id: string | null
}

export interface CreatedTicket {
  ticket_id: string
  requester_id: string
  title: string
  description: string
  category: string
  priority: string
  related_order_id: string | null
  created_at: string
}

export interface ConsoleAgentResponse {
  reply: string
  conversation_id: string
  trace_id: string
  route: string
  citations: RagCitation[]
  suggestions: string[]
  pending_ticket_confirmation: ConsoleAgentTicketConfirmation | null
  created_ticket: CreatedTicket | null
  human_handoff: ConsoleAgentHumanHandoff | null
  emotion: string | null
}

export interface ConsoleAgentConversationMessage {
  id: string
  role: ChatRole
  content: string
  created_at: string
  trace_id: string | null
  route: string | null
  citations: RagCitation[]
  suggestions: string[]
  pending_ticket_confirmation: ConsoleAgentTicketConfirmation | null
  created_ticket: CreatedTicket | null
  human_handoff: ConsoleAgentHumanHandoff | null
  emotion: string | null
  feedback_rating: 'helpful' | 'unhelpful' | null
  feedback_reason: ConsoleAgentFeedbackReason | null
}

export type ConsoleAgentFeedbackReason =
  | 'answer_incorrect'
  | 'intent_misunderstood'
  | 'citation_irrelevant'
  | 'should_handoff'
  | 'ticket_flow_incorrect'
  | 'other'

export interface ConsoleAgentFeedbackResponse {
  feedback_id: number
  rating: 'helpful' | 'unhelpful'
  reason: ConsoleAgentFeedbackReason | null
}

export interface ConsoleAgentConversationSummary {
  conversation_id: string
  title: string
  updated_at: string
}

export interface ConsoleAgentConversation extends ConsoleAgentConversationSummary {
  messages: ConsoleAgentConversationMessage[]
}

export interface ConsoleAgentStreamStage {
  stage: string
  label: string
}

export interface ConsoleAgentStreamDetail {
  kind: 'retrieval' | 'tool' | 'model'
  kb?: string
  hits?: number
  tool?: string
  order_id?: string
  model?: string
}

export interface RagCitation {
  source_index: number
  source: string
  title: string | null
  section: string | null
  chunk_id: string
  score: number
}

export interface ConsoleRagResponse {
  answer: string
  status: 'answered' | 'no_context'
  citations: RagCitation[]
  suggestions: string[]
  retrieved_count: number
  reranked_count: number
  used_rerank_fallback: boolean
  rerank_elapsed_ms: number
  collection_name: string
  embedding_model: string
  rerank_model: string
  llm_model: string
  trace_id: string
}

interface AiServiceErrorBody {
  code?: string
  message?: string
  trace_id?: string
}

export async function sendConsoleChatMessage(params: {
  message: string
  conversationId?: string
  history: ChatHistoryMessage[]
}) {
  try {
    const response = await aiApi.post<ConsoleChatResponse>('/api/ai/chat', {
      message: params.message,
      conversation_id: params.conversationId,
      history: params.history,
    })
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, 'AI 服务请求失败')
  }
}

export async function sendConsoleAgentMessage(params: {
  message: string
  conversationId?: string
}) {
  try {
    const response = await aiApi.post<ConsoleAgentResponse>('/api/ai/agent/conversations', {
      message: params.message,
      conversation_id: params.conversationId,
    })
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, 'AI 客服请求失败')
  }
}

export async function streamConsoleAgentMessage(
  params: {
    message: string
    conversationId?: string
  },
  onStage: (stage: ConsoleAgentStreamStage) => void,
  onDetail?: (detail: ConsoleAgentStreamDetail) => void,
): Promise<ConsoleAgentResponse> {
  const commonHeaders = aiApi.defaults.headers.common as Record<string, unknown>
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }
  for (const headerName of ['Authorization', 'X-Trace-Id']) {
    const value = commonHeaders[headerName]
    if (typeof value === 'string' && value) {
      headers[headerName] = value
    }
  }

  const response = await fetch(
    `${String(aiApi.defaults.baseURL || '').replace(/\/$/, '')}/api/ai/agent/conversations/stream`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify({
        message: params.message,
        conversation_id: params.conversationId,
      }),
    },
  )
  if (!response.ok || !response.body) {
    throw new Error(`AI 客服流式请求失败（HTTP ${response.status}）`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: ConsoleAgentResponse | undefined

  while (true) {
    const chunk = await reader.read()
    if (chunk.done) {
      break
    }
    buffer += decoder.decode(chunk.value, { stream: true })
    const events = buffer.replace(/\r\n/g, '\n').split('\n\n')
    buffer = events.pop() || ''

    for (const eventBlock of events) {
      const parsed = parseServerSentEvent(eventBlock)
      if (!parsed) {
        continue
      }
      if (parsed.event === 'stage') {
        onStage(parsed.data as ConsoleAgentStreamStage)
      } else if (parsed.event === 'detail') {
        onDetail?.(parsed.data as ConsoleAgentStreamDetail)
      } else if (parsed.event === 'result') {
        result = parsed.data as ConsoleAgentResponse
      } else if (parsed.event === 'error') {
        const error = parsed.data as { message?: string }
        throw new Error(error.message || 'AI 客服请求失败')
      }
    }
  }

  if (!result) {
    throw new Error('AI 客服流式响应未返回最终结果')
  }
  return result
}

export async function decideConsoleAgentTicketConfirmation(params: {
  conversationId: string
  confirmationId: string
  approved: boolean
}) {
  try {
    const response = await aiApi.post<ConsoleAgentResponse>(
      `/api/ai/agent/conversations/${encodeURIComponent(params.conversationId)}/confirmations/${encodeURIComponent(params.confirmationId)}`,
      { approved: params.approved },
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '工单确认操作失败')
  }
}

export async function requestConsoleAgentHumanHandoff(conversationId: string) {
  try {
    const response = await aiApi.post<ConsoleAgentResponse>(
      `/api/ai/agent/conversations/${encodeURIComponent(conversationId)}/human-handoff`,
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '无法发起人工转接')
  }
}

export async function listConsoleAgentConversations(limit = 20) {
  try {
    const response = await aiApi.get<ConsoleAgentConversationSummary[]>('/api/ai/agent/conversations', {
      params: { limit },
    })
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '无法读取最近会话')
  }
}

export async function getConsoleAgentConversation(conversationId: string) {  try {
    const response = await aiApi.get<ConsoleAgentConversation>(
      `/api/ai/agent/conversations/${encodeURIComponent(conversationId)}/history`,
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '无法恢复会话记录')
  }
}

export async function humanReply(conversationId: string, content: string): Promise<void> {
  try {
    await aiApi.post<ApiResponse<unknown>>(
      `/api/ai/agent/conversations/${encodeURIComponent(conversationId)}/human-reply`,
      { content },
    )
  } catch (error: any) {
    throw buildAiServiceError(error, '人工回复发送失败')
  }
}

export async function submitConsoleAgentFeedback(params: {
  conversationId: string
  traceId: string
  rating: 'helpful' | 'unhelpful'
  reason?: ConsoleAgentFeedbackReason
}) {
  try {
    const response = await aiApi.post<ConsoleAgentFeedbackResponse>(
      `/api/ai/agent/conversations/${encodeURIComponent(params.conversationId)}/feedback`,
      {
        trace_id: params.traceId,
        rating: params.rating,
        reason: params.reason,
      },
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '反馈提交失败')
  }
}

export async function correctConsoleAgentTicketConfirmation(params: {
  conversationId: string
  confirmationId: string
  ticketFields: ConsoleAgentTicketFields
}) {
  try {
    const response = await aiApi.put<ConsoleAgentResponse>(
      `/api/ai/agent/conversations/${encodeURIComponent(params.conversationId)}/confirmations/${encodeURIComponent(params.confirmationId)}`,
      { ticket_fields: params.ticketFields },
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '工单草稿修改失败')
  }
}

export async function sendConsoleRagMessage(params: {
  query: string
  candidateCount?: number
  topN?: number
}) {
  try {
    const response = await aiApi.post<ConsoleRagResponse>('/api/ai/rag/ask', {
      query: params.query,
      candidate_count: params.candidateCount || 20,
      top_n: params.topN || 5,
    })
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, 'RAG 知识库问答请求失败')
  }
}

function buildAiServiceError(error: any, fallbackMessage: string) {
  const data = error?.response?.data as AiServiceErrorBody | ApiResponse<unknown> | undefined
  if (data && typeof data === 'object' && 'message' in data && typeof data.message === 'string') {
    return new Error(data.message)
  }
  return new Error(fallbackMessage)
}

function parseServerSentEvent(eventBlock: string): { event: string; data: unknown } | undefined {
  let eventName = 'message'
  const dataLines: string[] = []
  for (const line of eventBlock.split('\n')) {
    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim()
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim())
    }
  }
  if (!dataLines.length) {
    return undefined
  }
  try {
    return { event: eventName, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return undefined
  }
}

export interface OpsStatsSummary {
  feedback: { helpful: number; unhelpful: number; helpful_rate: number | null }
  handoffs: { pending: number; in_progress: number; closed: number; total: number }
  emotion_distribution: Record<string, number>
  conversation_volume: Record<string, number>
}

export async function getOpsStatsSummary(days: number): Promise<OpsStatsSummary> {
  try {
    const response = await aiApi.get<OpsStatsSummary>('/api/ai/ops-stats/summary', {
      params: { days },
    })
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '无法读取运营统计数据')
  }
}

export interface ConversationSummary {
  summary: string
  source: 'llm' | 'rule'
}

export async function getConversationSummary(conversationId: string): Promise<ConversationSummary> {
  try {
    const response = await aiApi.get<ConversationSummary>(
      `/api/ai/agent/conversations/${encodeURIComponent(conversationId)}/summary`,
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '无法生成会话摘要')
  }
}
