<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { cancelOrder, listOrders, refundOrder } from '../services/businessApi'
import type { OrderListItem } from '../services/businessApi'

const orders = ref<OrderListItem[]>([])
const loading = ref(false)

const orderStatusLabels: Record<string, string> = {
  shipped: '运输中',
  waiting_shipment: '待发货',
  delivered: '已签收',
  canceled: '已取消',
}

const paymentStatusLabels: Record<string, string> = {
  paid: '已支付',
  unpaid: '未支付',
  refunded: '已退款',
}

async function loadOrders() {
  loading.value = true
  try {
    orders.value = await listOrders()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '订单加载失败')
  } finally {
    loading.value = false
  }
}

function formatDate(value: string | null | undefined) {
  return value ? value.replace('T', ' ').slice(0, 19) : '-'
}

async function handleCancel(row: OrderListItem) {
  try {
    const { value } = await ElMessageBox.prompt(
      '请输入取消原因，提交后将不可撤销',
      `取消订单 - ${row.order_id}`,
      {
        confirmButtonText: '确认取消',
        cancelButtonText: '取消',
        inputPlaceholder: '取消原因（必填）',
        inputValidator: (input: string) => {
          if (!input || !input.trim()) return '取消原因不能为空'
          return input.trim().length <= 200 ? true : '取消原因不能超过 200 字'
        },
      },
    )
    if (!value || !value.trim()) return
    await cancelOrder(row.order_id, value.trim())
    ElMessage.success('订单已取消')
    await loadOrders()
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '取消订单失败')
  }
}

async function handleRefund(row: OrderListItem) {
  try {
    const { value } = await ElMessageBox.prompt(
      '请输入退款原因，提交后将不可撤销',
      `申请退款 - ${row.order_id}`,
      {
        confirmButtonText: '确认退款',
        cancelButtonText: '取消',
        inputPlaceholder: '退款原因（必填）',
        inputValidator: (input: string) => {
          if (!input || !input.trim()) return '退款原因不能为空'
          return input.trim().length <= 100 ? true : '退款原因不能超过 100 字'
        },
      },
    )
    if (!value || !value.trim()) return
    await refundOrder(row.order_id, value.trim())
    ElMessage.success('退款成功')
    await loadOrders()
  } catch (error) {
    if (error === 'cancel' || error === 'close') return
    ElMessage.error(error instanceof Error ? error.message : '退款失败')
  }
}

onMounted(loadOrders)
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>我的订单</span>
        <el-button type="primary" plain :loading="loading" @click="loadOrders">刷新订单</el-button>
      </div>
    </template>

    <el-table v-loading="loading" :data="orders" stripe>
      <el-table-column prop="order_id" label="订单号" width="140" />
      <el-table-column prop="owner_user_id" label="用户" width="120" />
      <el-table-column label="订单状态" width="220">
        <template #default="{ row }">
          <template v-if="row.order_status === 'canceled'">
            <el-tag type="info" effect="light">已取消（{{ formatDate(row.canceled_at) }}）</el-tag>
            <div class="cancel-reason">取消原因：{{ row.cancel_reason || '-' }}</div>
          </template>
          <el-tag v-else effect="light">{{ orderStatusLabels[row.order_status] || row.order_status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="支付状态" width="200">
        <template #default="{ row }">
          <template v-if="row.payment_status === 'refunded'">
            已退款 ¥{{ row.refund_amount ?? '-' }}（{{ formatDate(row.refunded_at) }}）
          </template>
          <template v-else>{{ paymentStatusLabels[row.payment_status] || row.payment_status }}</template>
        </template>
      </el-table-column>
      <el-table-column prop="logistics_message" label="物流状态" min-width="220" show-overflow-tooltip />
      <el-table-column prop="latest_event" label="最新事件" min-width="220" show-overflow-tooltip />
      <el-table-column label="可建工单" width="100">
        <template #default="{ row }">
          <el-tag :type="row.can_create_ticket ? 'success' : 'info'" effect="light">
            {{ row.can_create_ticket ? '可以' : '不可' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="更新时间" width="180">
        <template #default="{ row }">{{ formatDate(row.updated_at) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="230" fixed="right">
        <template #default="{ row }">
          <template
            v-if="row.order_status === 'waiting_shipment' && row.payment_status !== 'refunded' && row.order_status !== 'canceled'"
          >
            <el-button type="primary" plain size="small" @click="handleRefund(row)">申请退款</el-button>
            <el-button type="danger" plain size="small" @click="handleCancel(row)">取消订单</el-button>
          </template>
          <span v-else>-</span>
        </template>
      </el-table-column>
    </el-table>

    <el-empty v-if="!loading && orders.length === 0" description="暂无可见订单" />
  </el-card>
</template>

<style scoped>
.cancel-reason {
  margin-top: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
  line-height: 1.4;
}
</style>
