import { aiApi, javaApi } from './http'
import type { ApiResponse } from './http'

export interface AiFeedbackReasonCount {
  reason: string
  count: number
}

export interface AiFeedbackRegressionCandidate {
  feedback_id: number
  conversation_id: string
  trace_id: string
  reason: string | null
  agent_route: string
  citation_count: number
  human_handoff_suggested: boolean
  review_status: string
  bad_case_id: string | null
  review_note: string | null
  created_at: string
}

export interface ProductionFeedbackContext extends AiFeedbackRegressionCandidate {
  user_message_excerpt: string | null
  assistant_answer_excerpt: string | null
  citation_summary: Array<{ source: string | null; title: string | null; chunk_id: string | null }>
}

export type RegressionAssertion =
  | 'intent'
  | 'citation_present'
  | 'ticket_confirmation_required'
  | 'tool_called'
  | 'must_ask_for'
  | 'must_not_reveal'

export type RegressionExpectedIntent =
  | 'policy_question'
  | 'order_query'
  | 'ticket_request'
  | 'refund_request'
  | 'smalltalk'
  | 'unsupported'
  | 'unclear'

export interface PromoteProductionFeedbackPayload {
  failure_layer: string
  severity: string
  failure_category: string
  expected_behavior: string
  recommended_action: string
  regression_action: string
  review_note: string
  regression_message: string
  regression_assertion: RegressionAssertion
  regression_expected_intent: RegressionExpectedIntent | null
  regression_expected_tool?: string
  regression_must_ask_fields?: string[]
  regression_must_not_reveal_terms?: string[]
}

export interface PromoteProductionFeedbackResponse {
  bad_case: { id: string; title: string; status: string }
  regression_draft: { suggested_case_id?: string; assertions?: string[] }
}

export interface AiResponseFeedbackOverview {
  total_count: number
  helpful_count: number
  unhelpful_count: number
  unhelpful_rate: number
  reason_counts: AiFeedbackReasonCount[]
  regression_candidates: AiFeedbackRegressionCandidate[]
}

export async function getAiResponseFeedbackOverview() {
  const response = await javaApi.get<ApiResponse<AiResponseFeedbackOverview>>('/api/ai-response-feedback/overview')
  if (!response.data.success) {
    throw new Error(response.data.message || '线上反馈加载失败')
  }
  return response.data.data
}

export async function getProductionFeedbackContext(feedbackId: number) {
  const response = await aiApi.get<ProductionFeedbackContext>(
    `/api/ai/evaluation/feedback-candidates/${feedbackId}`,
  )
  return response.data
}

export async function promoteProductionFeedback(feedbackId: number, payload: PromoteProductionFeedbackPayload) {
  const response = await aiApi.post<PromoteProductionFeedbackResponse>(
    `/api/ai/evaluation/feedback-candidates/${feedbackId}/promote`,
    payload,
  )
  return response.data
}

export async function reviewProductionFeedback(feedbackId: number, payload: { review_status: 'triaged' | 'closed'; review_note: string }) {
  const response = await aiApi.post<ProductionFeedbackContext>(
    `/api/ai/evaluation/feedback-candidates/${feedbackId}/review`,
    payload,
  )
  return response.data
}
