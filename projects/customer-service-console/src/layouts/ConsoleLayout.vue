<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { Component } from 'vue'
import {
  ChatDotRound,
  DataAnalysis,
  Document,
  HomeFilled,
  List,
  Operation,
  Setting,
  Tickets,
} from '@element-plus/icons-vue'
import type { UserRole } from '../stores/session'
import { useSessionStore } from '../stores/session'

const route = useRoute()
const router = useRouter()
const session = useSessionStore()

const activePath = computed(() => route.path)
const pageTitle = computed(() => String(route.meta.title || '运营概览'))

interface MenuItem {
  path: string
  label: string
  icon: Component
  roles?: UserRole[]
}

const menuItems: MenuItem[] = [
  { path: '/dashboard', label: '运营概览', icon: HomeFilled, roles: ['agent', 'supervisor', 'admin'] },
  { path: '/ai-chat', label: 'AI 客服', icon: ChatDotRound, roles: ['customer', 'agent', 'supervisor', 'admin'] },
  { path: '/orders', label: '我的订单', icon: List, roles: ['customer', 'agent', 'supervisor', 'admin'] },
  { path: '/tickets', label: '我的工单', icon: Tickets, roles: ['customer', 'agent', 'supervisor', 'admin'] },
  { path: '/workbench', label: '工单工作台', icon: Operation, roles: ['agent', 'supervisor', 'admin'] },
  { path: '/knowledge', label: '知识库管理', icon: Document, roles: ['supervisor', 'admin'] },
  { path: '/evaluation', label: 'AI 评估', icon: DataAnalysis, roles: ['supervisor', 'admin'] },
  { path: '/settings', label: '系统配置', icon: Setting, roles: ['admin'] },
]

const visibleMenuItems = computed(() => {
  const role = session.currentUser?.role
  return menuItems.filter((item) => !item.roles || (role && item.roles.includes(role)))
})

function handleLogout() {
  session.logout()
  router.push('/login')
}
</script>

<template>
  <el-container class="console-shell">
    <el-aside width="248px" class="console-sidebar">
      <div class="brand">
        <div class="brand-mark">AI</div>
        <div>
          <strong>智能工单系统</strong>
          <span>Customer Service Console</span>
        </div>
      </div>

      <el-menu :default-active="activePath" router class="console-menu">
        <el-menu-item v-for="item in visibleMenuItems" :key="item.path" :index="item.path">
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="console-header">
        <div>
          <p class="section-label">阶段 11 项目化</p>
          <h1>{{ pageTitle }}</h1>
        </div>
        <div class="user-panel">
          <el-tag type="success" effect="light">{{ session.roleLabel }}</el-tag>
          <el-avatar :size="36">{{ session.currentUser?.name.slice(0, 1) }}</el-avatar>
          <div>
            <strong>{{ session.currentUser?.name }}</strong>
            <span>{{ session.currentUser?.id }}</span>
          </div>
          <el-button text @click="handleLogout">退出</el-button>
        </div>
      </el-header>

      <el-main class="console-main">
        <RouterView />
      </el-main>
    </el-container>
  </el-container>
</template>
