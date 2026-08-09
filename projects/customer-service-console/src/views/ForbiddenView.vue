<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <main class="empty-state-page">
    <el-result
      icon="warning"
      title="当前角色无权访问"
      sub-title="这个页面需要更高角色权限。后续真实登录接入 Java 后，会由后端继续做接口权限校验。"
    >
      <template #extra>
        <el-button type="primary" @click="router.push('/')">回到运营概览</el-button>
      </template>
    </el-result>
  </main>
</template>
