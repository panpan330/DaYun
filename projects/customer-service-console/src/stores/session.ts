import { defineStore } from 'pinia'
import { applyAuthToken, applyCurrentUserHeaders, javaApi } from '../services/http'
import type { ApiResponse } from '../services/http'

export type UserRole = 'customer' | 'agent' | 'supervisor' | 'admin'

export interface CurrentUser {
  id: string
  name: string
  role: UserRole
  tenantId: string
}

interface LoginForm {
  username: string
  password: string
}

interface LoginApiUser {
  tenant_id: string
  user_id: string
  username: string
  display_name: string
  roles: UserRole[]
  default_home_path: string
}

interface LoginApiResponse {
  token: string
  user: LoginApiUser
}

interface StoredSession {
  token: string
  currentUser: CurrentUser
}

const STORAGE_KEY = 'customer-service-console-session'
const userRoles: UserRole[] = ['customer', 'agent', 'supervisor', 'admin']

export function getDefaultHomePath(role: UserRole | undefined) {
  if (role === 'customer') {
    return '/ai-chat'
  }
  if (role === 'agent') {
    return '/workbench'
  }
  if (role === 'supervisor') {
    return '/knowledge'
  }
  return '/dashboard'
}

function isCurrentUser(value: unknown): value is CurrentUser {
  if (!value || typeof value !== 'object') {
    return false
  }
  const user = value as CurrentUser
  return (
    typeof user.id === 'string' &&
    typeof user.name === 'string' &&
    typeof user.tenantId === 'string' &&
    userRoles.includes(user.role)
  )
}

function isStoredSession(value: unknown): value is StoredSession {
  if (!value || typeof value !== 'object') {
    return false
  }
  const session = value as StoredSession
  return typeof session.token === 'string' && isCurrentUser(session.currentUser)
}

function readStoredSession(): StoredSession | null {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) {
    return null
  }
  try {
    const parsed = JSON.parse(raw)
    if (isStoredSession(parsed)) {
      return parsed
    }
    localStorage.removeItem(STORAGE_KEY)
    return null
  } catch {
    localStorage.removeItem(STORAGE_KEY)
    return null
  }
}

function toCurrentUser(apiUser: LoginApiUser): CurrentUser {
  const role = apiUser.roles.find((value) => userRoles.includes(value)) || 'customer'
  return {
    id: apiUser.user_id,
    name: apiUser.display_name,
    role,
    tenantId: apiUser.tenant_id,
  }
}

const initialSession = readStoredSession()
applyAuthToken(initialSession?.token ?? null)
applyCurrentUserHeaders(initialSession?.currentUser ?? null)

export const useSessionStore = defineStore('session', {
  state: () => ({
    currentUser: initialSession?.currentUser ?? null,
    token: initialSession?.token ?? null,
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.currentUser && state.token),
    roleLabel: (state) => {
      const labels: Record<UserRole, string> = {
        customer: '客户',
        agent: '客服',
        supervisor: '主管',
        admin: '管理员',
      }
      return state.currentUser ? labels[state.currentUser.role] : '未登录'
    },
  },
  actions: {
    async login(form: LoginForm) {
      const response = await javaApi.post<ApiResponse<LoginApiResponse>>('/api/auth/login', {
        username: form.username.trim(),
        password: form.password,
        tenant_id: 'default',
      })

      if (!response.data.success) {
        throw new Error(response.data.message || '登录失败')
      }

      const user = toCurrentUser(response.data.data.user)
      const token = response.data.data.token
      this.currentUser = user
      this.token = token
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ token, currentUser: user }))
      applyAuthToken(token)
      applyCurrentUserHeaders(user)
      return user
    },
    logout() {
      this.currentUser = null
      this.token = null
      localStorage.removeItem(STORAGE_KEY)
      applyAuthToken(null)
      applyCurrentUserHeaders(null)
    },
  },
})
