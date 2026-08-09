import { createRouter, createWebHistory } from 'vue-router'
import type { UserRole } from '../stores/session'
import { getDefaultHomePath, useSessionStore } from '../stores/session'
import ConsoleLayout from '../layouts/ConsoleLayout.vue'
import LoginView from '../views/LoginView.vue'
import DashboardView from '../views/DashboardView.vue'
import AiChatView from '../views/AiChatView.vue'
import OrdersView from '../views/OrdersView.vue'
import TicketsView from '../views/TicketsView.vue'
import TicketWorkbenchView from '../views/TicketWorkbenchView.vue'
import KnowledgeBaseView from '../views/KnowledgeBaseView.vue'
import EvaluationView from '../views/EvaluationView.vue'
import SettingsView from '../views/SettingsView.vue'
import ForbiddenView from '../views/ForbiddenView.vue'
import NotFoundView from '../views/NotFoundView.vue'

declare module 'vue-router' {
  interface RouteMeta {
    title?: string
    public?: boolean
    roles?: UserRole[]
  }
}

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: LoginView, meta: { title: '登录', public: true } },
    { path: '/403', name: 'forbidden', component: ForbiddenView, meta: { title: '无权限' } },
    {
      path: '/',
      component: ConsoleLayout,
      children: [
        { path: '', name: 'home', component: DashboardView, meta: { title: '首页' } },
        { path: 'dashboard', name: 'dashboard', component: DashboardView, meta: { title: '运营概览', roles: ['agent', 'supervisor', 'admin'] } },
        { path: 'ai-chat', name: 'ai-chat', component: AiChatView, meta: { title: 'AI 客服', roles: ['customer', 'agent', 'supervisor', 'admin'] } },
        { path: 'orders', name: 'orders', component: OrdersView, meta: { title: '我的订单', roles: ['customer', 'agent', 'supervisor', 'admin'] } },
        { path: 'tickets', name: 'tickets', component: TicketsView, meta: { title: '我的工单', roles: ['customer', 'agent', 'supervisor', 'admin'] } },
        { path: 'workbench', name: 'workbench', component: TicketWorkbenchView, meta: { title: '工单工作台', roles: ['agent', 'supervisor', 'admin'] } },
        { path: 'knowledge', name: 'knowledge', component: KnowledgeBaseView, meta: { title: '知识库管理', roles: ['supervisor', 'admin'] } },
        { path: 'evaluation', name: 'evaluation', component: EvaluationView, meta: { title: 'AI 评估', roles: ['supervisor', 'admin'] } },
        { path: 'settings', name: 'settings', component: SettingsView, meta: { title: '系统配置', roles: ['admin'] } },
      ],
    },
    { path: '/:pathMatch(.*)*', name: 'not-found', component: NotFoundView, meta: { title: '页面不存在' } },
  ],
})

router.beforeEach((to) => {
  const session = useSessionStore()

  if (to.meta.public) {
    if (to.name === 'login' && session.isAuthenticated) {
      return getDefaultHomePath(session.currentUser?.role)
    }
    return true
  }

  if (!session.isAuthenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  if (to.name === 'home') {
    return getDefaultHomePath(session.currentUser?.role)
  }

  const allowedRoles = to.meta.roles
  if (allowedRoles && session.currentUser && !allowedRoles.includes(session.currentUser.role)) {
    return { name: 'forbidden' }
  }

  return true
})
