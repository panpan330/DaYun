import { javaApi } from './http'
import type { ApiResponse } from './http'

export interface OrderListItem {
  order_id: string
  owner_user_id: string
  order_status: string
  payment_status: string
  logistics_message: string
  latest_event: string
  can_create_ticket: boolean
  refund_amount: number | null
  refunded_at: string | null
  canceled_at: string | null
  cancel_reason: string | null
  updated_at: string
}

export interface OrderToolView {
  order_id: string
  order_status: string
  payment_status: string
  logistics_message: string
  latest_event: string
  can_create_ticket: boolean
  refund_amount: number | null
  refunded_at: string | null
  refund_reason: string | null
  canceled_at: string | null
  cancel_reason: string | null
  user_visible_summary: string | null
}

export interface TicketListItem {
  ticket_id: string
  requester_user_id: string
  ticket_status: string
  title: string
  category: string
  priority: string
  related_order_id: string | null
  assignee_user_id: string | null
  assignee_display_name: string | null
  source: string
  created_at: string
  updated_at: string
}

export interface TicketEventItem {
  event_id: string
  event_type: string
  event_payload: string
  operator_type: string
  operator_id: string
  trace_id: string
  created_at: string
}

export interface TicketMessageItem {
  message_id: string
  visibility: 'public' | 'internal'
  content: string
  author_type: string
  author_user_id: string
  author_display_name: string
  created_at: string
}

export interface TicketDetail extends TicketListItem {
  description: string
  created_trace_id: string
  events: TicketEventItem[]
  messages: TicketMessageItem[]
}

export interface UpdateTicketStatusPayload {
  target_status: 'in_progress' | 'waiting_user' | 'closed'
  note?: string
}

export interface AssignTicketPayload {
  assignee_user_id: string
  note?: string
}

export interface AddTicketMessagePayload {
  visibility: 'public' | 'internal'
  content: string
}

export interface TicketResolutionPayload {
  content: string
}

export interface ReopenTicketPayload {
  content: string
}

export interface StaffUserItem {
  user_id: string
  username: string
  display_name: string
  roles: string[]
}

export interface KnowledgeDocumentItem {
  document_id: string
  title: string
  doc_type: string
  business_domain: string
  permission_group: string
  status: string
  source_file_name: string
  chunk_count: number
  updated_by: string
  updated_at: string
}

function unwrapResponse<T>(response: ApiResponse<T>): T {
  if (!response.success) {
    throw new Error(response.message || response.code || '请求失败')
  }
  return response.data
}

export async function listOrders() {
  const response = await javaApi.get<ApiResponse<OrderListItem[]>>('/api/orders')
  return unwrapResponse(response.data)
}

export async function refundOrder(orderId: string, reason: string) {
  const response = await javaApi.post<ApiResponse<OrderToolView>>(`/api/orders/${orderId}/refund`, { reason })
  return unwrapResponse(response.data)
}

export async function cancelOrder(orderId: string, reason: string) {
  const response = await javaApi.post<ApiResponse<OrderToolView>>(`/api/orders/${orderId}/cancel`, { reason })
  return unwrapResponse(response.data)
}

export async function listTickets() {
  const response = await javaApi.get<ApiResponse<TicketListItem[]>>('/api/tickets')
  return unwrapResponse(response.data)
}

export async function getTicketDetail(ticketId: string) {
  const response = await javaApi.get<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}`)
  return unwrapResponse(response.data)
}

export async function updateTicketStatus(ticketId: string, payload: UpdateTicketStatusPayload) {
  const response = await javaApi.patch<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/status`, payload)
  return unwrapResponse(response.data)
}

export async function claimTicket(ticketId: string) {
  const response = await javaApi.patch<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/assignment/claim`)
  return unwrapResponse(response.data)
}

export async function assignTicket(ticketId: string, payload: AssignTicketPayload) {
  const response = await javaApi.patch<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/assignment`, payload)
  return unwrapResponse(response.data)
}

export async function addTicketMessage(ticketId: string, payload: AddTicketMessagePayload) {
  const response = await javaApi.post<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/messages`, payload)
  return unwrapResponse(response.data)
}

export async function resolveTicket(ticketId: string, payload: TicketResolutionPayload) {
  const response = await javaApi.post<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/resolution`, payload)
  return unwrapResponse(response.data)
}

export async function reopenTicket(ticketId: string, payload: ReopenTicketPayload) {
  const response = await javaApi.post<ApiResponse<TicketDetail>>(`/api/tickets/${ticketId}/reopen`, payload)
  return unwrapResponse(response.data)
}

export async function listStaffUsers() {
  const response = await javaApi.get<ApiResponse<StaffUserItem[]>>('/api/users/staff')
  return unwrapResponse(response.data)
}

export async function listKnowledgeDocuments() {
  const response = await javaApi.get<ApiResponse<KnowledgeDocumentItem[]>>('/api/knowledge-documents')
  return unwrapResponse(response.data)
}

export interface HumanHandoff {
  id: number
  conversation_id: string
  user_id: string
  reason: string
  related_order_id: string | null
  emotion: string | null
  status: 'pending' | 'in_progress' | 'closed'
  assigned_agent: string | null
  note: string | null
  created_at: string
  resolved_at: string | null
}

export async function listHumanHandoffs(status?: string) {
  const response = await javaApi.get<ApiResponse<HumanHandoff[]>>('/api/human-handoffs', {
    params: status ? { status } : {},
  })
  return unwrapResponse(response.data)
}

export async function claimHumanHandoff(id: number) {
  const response = await javaApi.post<ApiResponse<HumanHandoff>>(`/api/human-handoffs/${id}/claim`)
  return unwrapResponse(response.data)
}

export async function closeHumanHandoff(id: number, note: string) {
  const response = await javaApi.post<ApiResponse<HumanHandoff>>(`/api/human-handoffs/${id}/close`, {
    note,
  })
  return unwrapResponse(response.data)
}

export async function transferHumanHandoff(id: number, targetAgent: string) {
  const response = await javaApi.post<ApiResponse<HumanHandoff>>(
    `/api/human-handoffs/${id}/transfer`,
    { target_agent: targetAgent },
  )
  return unwrapResponse(response.data)
}
