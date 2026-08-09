import axios from 'axios'
import type { CurrentUser } from '../stores/session'

export interface ApiResponse<T> {
  success: boolean
  code: string
  message: string
  data: T
  trace_id: string
}

export const javaApi = axios.create({
  baseURL: import.meta.env.VITE_JAVA_API_BASE_URL || 'http://127.0.0.1:18004',
  timeout: 10000,
})

export const aiApi = axios.create({
  baseURL: import.meta.env.VITE_AI_API_BASE_URL || 'http://127.0.0.1:8000',
  timeout: 30000,
})

export function applyTraceHeaders(traceId: string) {
  javaApi.defaults.headers.common['X-Trace-Id'] = traceId
  aiApi.defaults.headers.common['X-Trace-Id'] = traceId
}

export function applyAuthToken(token: string | null) {
  const clients = [javaApi, aiApi]
  for (const client of clients) {
    if (!token) {
      delete client.defaults.headers.common.Authorization
      continue
    }
    client.defaults.headers.common.Authorization = `Bearer ${token}`
  }
}

export function applyCurrentUserHeaders(user: CurrentUser | null) {
  const clients = [javaApi, aiApi]
  for (const client of clients) {
    if (!user) {
      delete client.defaults.headers.common['X-User-Id']
      delete client.defaults.headers.common['X-Tenant-Id']
      delete client.defaults.headers.common['X-User-Role']
      continue
    }
    client.defaults.headers.common['X-User-Id'] = user.id
    client.defaults.headers.common['X-Tenant-Id'] = user.tenantId
    client.defaults.headers.common['X-User-Role'] = user.role
  }
}
