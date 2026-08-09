<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()
</script>

<template>
  <main class="empty-state-page">
    <el-result
      icon="info-filled"
      title="页面不存在"
      sub-title="请检查访问路径，或回到工作台继续操作。"
    >
      <template #extra>
        <el-button type="primary" @click="router.push('/')">回到运营概览</el-button>
      </template>
    </el-result>
  </main>
</template>
