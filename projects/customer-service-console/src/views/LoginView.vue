<script setup lang="ts">
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getDefaultHomePath, useSessionStore } from '../stores/session'

const route = useRoute()
const router = useRouter()
const session = useSessionStore()
const loading = ref(false)

const form = reactive({
  username: 'admin',
  password: '123456',
})

const demoAccounts = [
  { username: 'customer', role: '客户', description: '查看自己的订单、工单和 AI 客服入口' },
  { username: 'agent', role: '客服', description: '处理工单工作台，不看系统配置' },
  { username: 'supervisor', role: '主管', description: '管理知识库、评估和工单队列' },
  { username: 'admin', role: '管理员', description: '查看全部菜单和系统配置摘要' },
]

async function handleLogin() {
  if (loading.value) {
    return
  }
  loading.value = true
  try {
    const user = await session.login(form)
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
    await router.push(redirect === '/' ? getDefaultHomePath(user.role) : redirect)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '登录失败')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <main class="login-page">
    <section class="login-intro">
      <p class="section-label">阶段 11 项目化</p>
      <h1>AI 客服与智能工单系统</h1>
      <p>
        登录已接入 Java 业务服务，本地账号由 MySQL 初始化数据提供。登录后前端会携带本地开发 token 调用订单、工单和知识库接口。
      </p>
      <div class="login-capabilities">
        <el-tag>Vue3</el-tag>
        <el-tag>Element Plus</el-tag>
        <el-tag>Pinia Session</el-tag>
        <el-tag>Java Auth API</el-tag>
      </div>
    </section>

    <el-card class="login-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>登录工作台</span>
          <el-tag type="success">Java 本地账号</el-tag>
        </div>
      </template>

      <el-form label-position="top" @submit.prevent="handleLogin">
        <el-form-item label="账号">
          <el-input v-model="form.username" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" autocomplete="current-password" show-password />
        </el-form-item>
        <el-button class="login-button" type="primary" :loading="loading" @click="handleLogin">登录</el-button>
      </el-form>

      <el-divider />

      <div class="demo-account-list">
        <button
          v-for="account in demoAccounts"
          :key="account.username"
          type="button"
          class="demo-account"
          @click="form.username = account.username"
        >
          <strong>{{ account.username }} / {{ account.role }}</strong>
          <span>{{ account.description }}</span>
        </button>
      </div>
      <p class="login-hint">所有本地演示账号密码都是 123456。</p>
    </el-card>
  </main>
</template>
