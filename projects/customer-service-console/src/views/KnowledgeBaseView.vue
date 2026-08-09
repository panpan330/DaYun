<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  createKnowledgeBaseDocument,
  deleteKnowledgeBaseDocument,
  getKnowledgeBaseCollections,
  getKnowledgeBaseStatus,
  ingestKnowledgeBase,
  ingestKnowledgeBaseDocument,
  listKnowledgeBaseDocuments,
  updateKnowledgeBaseDocument,
  uploadKnowledgeBaseDocument,
} from '../services/knowledgeBaseApi'
import type {
  KnowledgeBaseCollectionStatus,
  KnowledgeBaseCollectionsResponse,
  KnowledgeBaseDocumentItem,
  KnowledgeBaseDocumentStatus,
  KnowledgeBaseEmbeddingMode,
  KnowledgeBaseIngestResponse,
  KnowledgeBaseStatusResponse,
} from '../services/knowledgeBaseApi'

const documents = ref<KnowledgeBaseDocumentItem[]>([])
const kbStatus = ref<KnowledgeBaseStatusResponse>()
const lastIngestResult = ref<KnowledgeBaseIngestResponse>()
const loading = ref(false)
const ingestingMode = ref<KnowledgeBaseEmbeddingMode | null>(null)
const collections = ref<KnowledgeBaseCollectionStatus[]>([])
const collectionsLoading = ref(false)
const dialogVisible = ref(false)
const dialogMode = ref<'create' | 'edit'>('create')
const dialogSubmitting = ref(false)
const editingDocumentId = ref('')
const dialogForm = ref({
  document_id: '',
  title: '',
  content: '',
  business_domain: 'general',
  permission_group: 'public',
  doc_type: 'policy',
  collection_name: 'kb_customer_policy',
  embedding_mode: 'fake' as KnowledgeBaseEmbeddingMode,
  chunk_size: 500,
  chunk_overlap: 80,
  file: null as File | null,
})

const localDocumentBySource = computed(() => {
  const result = new Map<string, KnowledgeBaseDocumentStatus>()
  for (const document of kbStatus.value?.documents || []) {
    result.set(document.source, document)
  }
  return result
})

const statusTagType: Record<string, 'success' | 'warning' | 'info' | 'danger'> = {
  active: 'success',
  draft: 'warning',
  disabled: 'info',
  failed: 'danger',
}

const domainLabels: Record<string, string> = {
  logistics: '物流',
  refund: '退款',
  account: '账号',
  order: '订单',
  general: '通用',
}

const permissionLabels: Record<string, string> = {
  public: '公开',
  customer: '客户可见',
  customer_service: '客服可见',
  internal: '内部',
}

function handleFileChange(file: { raw?: File; name: string; size: number }) {
  dialogForm.value.file = file.raw ?? null
}

function handleFileRemove() {
  dialogForm.value.file = null
}

async function loadPageData() {
  loading.value = true
  try {
    const [status, documentList] = await Promise.all([
      getKnowledgeBaseStatus(),
      listKnowledgeBaseDocuments(),
    ])
    kbStatus.value = status
    documents.value = documentList.documents
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '知识库加载失败')
  } finally {
    loading.value = false
  }
}

async function runIngest(mode: KnowledgeBaseEmbeddingMode) {
  if (mode === 'real') {
    await ElMessageBox.confirm(
      '真实入库会调用真实 embedding 模型并消耗 API 额度，确认继续吗？',
      '确认真实入库',
      {
        confirmButtonText: '确认入库',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )
  }

  ingestingMode.value = mode
  try {
    const result = await ingestKnowledgeBase({
      embedding_mode: mode,
      refresh: true,
      wait: true,
      include_readme: false,
      chunk_size: 500,
      chunk_overlap: 80,
    })
    lastIngestResult.value = result
    ElMessage.success(`入库完成：${result.chunk_count} 个 chunk，${result.vector_count} 条向量`)
    await loadPageData()
  } catch (error) {
    if (error === 'cancel' || error === 'close') {
      return
    }
    if (error instanceof Error && error.message.includes('cancel')) {
      return
    }
    ElMessage.error(error instanceof Error ? error.message : '知识库入库失败')
  } finally {
    ingestingMode.value = null
  }
}

function hasLocalFile(row: KnowledgeBaseDocumentItem) {
  return localDocumentBySource.value.has(row.source_file_name)
}

function formatDate(value: string) {
  return value ? value.replace('T', ' ').slice(0, 19) : '-'
}

function labelOf(value: string, labels: Record<string, string>) {
  return labels[value] || value
}

function openCreateDialog() {
  dialogMode.value = 'create'
  editingDocumentId.value = ''
  dialogForm.value = {
    document_id: '',
    title: '',
    content: '',
    business_domain: 'general',
    permission_group: 'public',
    doc_type: 'policy',
    collection_name: 'kb_customer_policy',
    embedding_mode: 'fake',
    chunk_size: 500,
    chunk_overlap: 80,
    file: null,
  }
  dialogVisible.value = true
}

function openEditDialog(row: KnowledgeBaseDocumentItem) {
  dialogMode.value = 'edit'
  editingDocumentId.value = row.document_id
  dialogForm.value = {
    document_id: row.document_id,
    title: row.title,
    content: '',
    business_domain: row.business_domain,
    permission_group: row.permission_group,
    doc_type: row.doc_type,
    collection_name: row.collection_name || 'kb_customer_policy',
    embedding_mode: 'fake',
    chunk_size: 500,
    chunk_overlap: 80,
    file: null,
  }
  dialogVisible.value = true
}

async function submitDocumentDialog() {
  const form = dialogForm.value
  if (dialogMode.value === 'create' && !form.file) {
    if (!form.title.trim()) {
      ElMessage.warning('文档标题不能为空')
      return
    }
    if (!form.document_id.trim()) {
      ElMessage.warning('文档 ID 不能为空')
      return
    }
    if (!form.content.trim()) {
      ElMessage.warning('文档内容不能为空')
      return
    }
  }

  dialogSubmitting.value = true
  try {
    if (dialogMode.value === 'create') {
      if (form.file) {
        const formData = new FormData()
        formData.append('file', form.file)
        if (form.document_id.trim()) {
          formData.append('document_id', form.document_id.trim())
        }
        formData.append('business_domain', form.business_domain)
        formData.append('permission_group', form.permission_group)
        formData.append('embedding_mode', form.embedding_mode)
        formData.append('chunk_size', String(form.chunk_size))
        formData.append('chunk_overlap', String(form.chunk_overlap))
        await uploadKnowledgeBaseDocument(formData)
        ElMessage.success('文档上传并同步成功')
      } else {
        await createKnowledgeBaseDocument({
          document_id: form.document_id.trim(),
          title: form.title.trim(),
          content: form.content,
          business_domain: form.business_domain,
          permission_group: form.permission_group,
          doc_type: form.doc_type,
          collection_name: form.collection_name,
          embedding_mode: form.embedding_mode,
          chunk_size: form.chunk_size,
          chunk_overlap: form.chunk_overlap,
        })
        ElMessage.success('文档创建并同步成功')
      }
    } else {
      const updatePayload: Record<string, unknown> = {
        title: form.title.trim(),
        business_domain: form.business_domain,
        permission_group: form.permission_group,
        doc_type: form.doc_type,
        embedding_mode: form.embedding_mode,
        chunk_size: form.chunk_size,
        chunk_overlap: form.chunk_overlap,
      }
      if (form.content.trim()) {
        updatePayload.content = form.content
      }
      await updateKnowledgeBaseDocument(editingDocumentId.value, updatePayload)
      ElMessage.success('文档更新并重新同步成功')
    }
    dialogVisible.value = false
    await loadPageData()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '文档保存失败')
  } finally {
    dialogSubmitting.value = false
  }
}

async function confirmDeleteDocument(row: KnowledgeBaseDocumentItem) {
  try {
    await ElMessageBox.confirm(
      `确认删除文档「${row.title}」？本地文件、Qdrant chunk 与元数据将被一并删除。`,
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning',
      },
    )
  } catch {
    return
  }
  try {
    await deleteKnowledgeBaseDocument(row.document_id)
    ElMessage.success('文档已删除')
    await loadPageData()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '文档删除失败')
  }
}

async function runSingleIngest(row: KnowledgeBaseDocumentItem) {
  try {
    await ingestKnowledgeBaseDocument(row.document_id, {
      embedding_mode: 'fake',
      chunk_size: 500,
      chunk_overlap: 80,
    })
    ElMessage.success(`文档「${row.title}」同步完成`)
    await loadPageData()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '文档同步失败')
  }
}

async function loadCollections() {
  collectionsLoading.value = true
  try {
    const data: KnowledgeBaseCollectionsResponse = await getKnowledgeBaseCollections()
    collections.value = data.collections
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'Collection 列表加载失败')
  } finally {
    collectionsLoading.value = false
  }
}

onMounted(async () => {
  await loadPageData()
  await loadCollections()
})
</script>

<template>
  <section class="knowledge-page">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>知识库管理</span>
          <div class="toolbar-actions">
            <el-button plain :loading="loading" @click="loadPageData">刷新状态</el-button>
            <el-button type="success" plain :disabled="dialogSubmitting" @click="openCreateDialog">
              上传文档
            </el-button>
            <el-button
              type="primary"
              plain
              :loading="ingestingMode === 'fake'"
              :disabled="Boolean(ingestingMode)"
              @click="runIngest('fake')"
            >
              Fake 入库
            </el-button>
            <el-button
              type="primary"
              :loading="ingestingMode === 'real'"
              :disabled="Boolean(ingestingMode) || !kbStatus?.real_embedding_configured"
              @click="runIngest('real')"
            >
              真实入库
            </el-button>
          </div>
        </div>
      </template>

      <div class="kb-summary">
        <el-statistic title="业务文档" :value="documents.length" />
        <el-statistic title="本地知识文件" :value="kbStatus?.document_count || 0" />
        <el-statistic title="Fake 向量维度" :value="kbStatus?.fake_embedding_dimension || 0" />
        <el-statistic title="最近入库向量" :value="lastIngestResult?.vector_count || 0" />
      </div>

      <el-alert
        v-if="kbStatus && !kbStatus.real_embedding_configured"
        class="page-alert"
        type="warning"
        :closable="false"
        title="当前未配置真实 embedding API Key，只能使用 Fake 入库验证 Qdrant 链路。"
      />

      <el-descriptions v-if="kbStatus" class="page-alert" :column="3" border>
        <el-descriptions-item label="Qdrant 地址">{{ kbStatus.qdrant_base_url }}</el-descriptions-item>
        <el-descriptions-item label="Collection">{{ kbStatus.collection_name }}</el-descriptions-item>
        <el-descriptions-item label="trace_id">{{ kbStatus.trace_id }}</el-descriptions-item>
      </el-descriptions>

      <div v-loading="collectionsLoading" class="collection-status">
        <span class="collection-status-title">Collection 状态</span>
        <el-tag
          v-for="collection in collections"
          :key="collection.collection_name"
          class="collection-tag"
          :type="collection.exists ? 'success' : 'info'"
          effect="plain"
        >
          {{ collection.collection_name }}
          <template v-if="collection.exists">
            · {{ collection.point_count }} 点
          </template>
          <template v-else>
            · 未创建
          </template>
        </el-tag>
      </div>

      <el-table v-loading="loading" :data="documents" stripe>
        <el-table-column prop="title" label="文档名称" min-width="220" show-overflow-tooltip />
        <el-table-column label="业务域" width="120">
          <template #default="{ row }">
            {{ labelOf(row.business_domain, domainLabels) }}
          </template>
        </el-table-column>
        <el-table-column label="权限组" width="130">
          <template #default="{ row }">
            {{ labelOf(row.permission_group, permissionLabels) }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusTagType[row.status] || 'info'" effect="light">
              {{ row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="Chunks" width="100" />
        <el-table-column prop="source_file_name" label="源文件" min-width="190" show-overflow-tooltip />
        <el-table-column label="本地文件" width="110">
          <template #default="{ row }">
            <el-tag :type="hasLocalFile(row) ? 'success' : 'danger'" effect="light">
              {{ hasLocalFile(row) ? '已匹配' : '缺失' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="更新时间" width="180">
          <template #default="{ row }">{{ formatDate(row.updated_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="openEditDialog(row)">编辑</el-button>
            <el-button link type="primary" :disabled="Boolean(ingestingMode)" @click="runSingleIngest(row)">
              同步
            </el-button>
            <el-button link type="danger" @click="confirmDeleteDocument(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-empty v-if="!loading && documents.length === 0" description="暂无可见知识文档" />

      <el-dialog
        v-model="dialogVisible"
        :title="dialogMode === 'create' ? '上传知识文档' : '编辑知识文档'"
        width="640px"
        destroy-on-close
      >
        <el-form label-width="110px">
          <el-form-item label="上传文件">
            <el-upload
              :auto-upload="false"
              :limit="1"
              accept=".md,.txt,.pdf"
              :on-change="handleFileChange"
              :on-remove="handleFileRemove"
            >
              <el-button type="primary" plain>选择文件</el-button>
              <template #tip>
                <div class="el-upload__tip">支持 .md / .txt / .pdf；选择文件后将以文件内容入库</div>
              </template>
            </el-upload>
          </el-form-item>
          <el-form-item v-if="dialogMode === 'create'" label="文档 ID" required>
            <el-input v-model="dialogForm.document_id" placeholder="如 doc-001（唯一标识；不填则用文件名）" />
          </el-form-item>
          <el-form-item label="标题" required>
            <el-input v-model="dialogForm.title" placeholder="文档标题" />
          </el-form-item>
          <el-form-item label="内容" :required="dialogMode === 'create' && !dialogForm.file">
            <el-input
              v-model="dialogForm.content"
              type="textarea"
              :rows="8"
              :disabled="dialogForm.file !== null"
              :placeholder="dialogForm.file ? '已选择文件，将以文件内容入库' : (dialogMode === 'edit' ? '留空则保留原文档内容' : '文档正文（Markdown）')"
            />
          </el-form-item>
          <el-form-item label="业务域">
            <el-select v-model="dialogForm.business_domain">
              <el-option label="退款" value="refund" />
              <el-option label="物流" value="logistics" />
              <el-option label="订单" value="order" />
              <el-option label="账号" value="account" />
              <el-option label="通用" value="general" />
            </el-select>
          </el-form-item>
          <el-form-item label="权限组">
            <el-select v-model="dialogForm.permission_group">
              <el-option label="公开" value="public" />
              <el-option label="客户可见" value="customer" />
              <el-option label="客服可见" value="customer_service" />
              <el-option label="内部" value="internal" />
            </el-select>
          </el-form-item>
          <el-form-item label="Collection">
            <el-select v-model="dialogForm.collection_name">
              <el-option
                v-for="collection in collections"
                :key="collection.collection_name"
                :label="collection.collection_name"
                :value="collection.collection_name"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="向量模式">
            <el-select v-model="dialogForm.embedding_mode">
              <el-option label="Fake（离线）" value="fake" />
              <el-option label="真实（API）" value="real" />
            </el-select>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" :loading="dialogSubmitting" @click="submitDocumentDialog">
            {{ dialogMode === 'create' ? '上传并同步' : '保存并重新同步' }}
          </el-button>
        </template>
      </el-dialog>
    </el-card>
  </section>
</template>

<style scoped>
.knowledge-page {
  display: grid;
  gap: 16px;
}

.kb-summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
}

.kb-summary :deep(.el-statistic) {
  min-height: 88px;
  border: 1px solid #dce3ee;
  border-radius: 8px;
  padding: 14px 16px;
  background: #f8fafc;
}

.collection-status {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}

.collection-status-title {
  font-size: 13px;
  color: #606266;
  margin-right: 4px;
}

.collection-tag {
  margin-right: 0;
}

@media (max-width: 1100px) {
  .kb-summary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 760px) {
  .kb-summary {
    grid-template-columns: 1fr;
  }
}
</style>
