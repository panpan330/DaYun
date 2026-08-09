<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, shallowRef, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { ArrowDown } from '@element-plus/icons-vue'
import { marked } from 'marked'
import * as echarts from 'echarts'
import type { EChartsType } from 'echarts'
import {
  getEvaluationHistory,
  getEvaluationOverview,
  getLatestReport,
  getRagEvalRuns,
  runProductionRegression,
} from '../services/evaluationApi'
import {
  getAiResponseFeedbackOverview,
  getProductionFeedbackContext,
  promoteProductionFeedback,
  reviewProductionFeedback,
} from '../services/aiFeedbackApi'
import type {
  BadCaseItem,
  BaselineComparison,
  BaselineMetricComparison,
  EvaluationHistory,
  EvaluationHistoryPoint,
  EvaluationOverview,
  EvaluationReportType,
  RagEvalRun,
} from '../services/evaluationApi'
import type {
  AiFeedbackRegressionCandidate,
  AiResponseFeedbackOverview,
  ProductionFeedbackContext,
  PromoteProductionFeedbackPayload,
  RegressionAssertion,
  RegressionExpectedIntent,
} from '../services/aiFeedbackApi'

const overview = ref<EvaluationOverview | null>(null)
const loading = ref(false)
const selectedBadCaseId = ref('')
const feedbackOverview = ref<AiResponseFeedbackOverview | null>(null)
const feedbackReviewVisible = ref(false)
const feedbackReviewLoading = ref(false)
const feedbackPromotionSubmitting = ref(false)
const productionRegressionRunning = ref(false)
const trendHistory = ref<EvaluationHistory | null>(null)
const trendLoading = ref(false)
const previewVisible = ref(false)
const previewReport = ref<{ report: string; type: string; generated_at: string } | null>(null)
const previewHtml = ref('')

const trendChartEl = ref<HTMLDivElement | null>(null)
const trendChart = shallowRef<EChartsType | null>(null)

const regressionPieEl = ref<HTMLDivElement | null>(null)
const regressionPie = shallowRef<EChartsType | null>(null)

const ragEvalRuns = ref<RagEvalRun[]>([])
const ragEvalLoading = ref(false)
const ragEvalChartEl = ref<HTMLDivElement | null>(null)
const ragEvalChart = shallowRef<EChartsType | null>(null)
const reportDownloading = ref(false)
const selectedFeedbackContext = ref<ProductionFeedbackContext | null>(null)
const feedbackPromotionForm = reactive({
  failure_layer: 'agent_decision',
  severity: 'medium',
  failure_category: '',
  expected_behavior: '',
  recommended_action: '',
  regression_action: '',
  review_note: '',
  regression_message: '',
  regression_assertion: 'intent' as RegressionAssertion,
  regression_expected_intent: null as RegressionExpectedIntent | null,
  regression_expected_tool: '',
  regression_must_ask_fields: '',
  regression_must_not_reveal_terms: '',
})

const selectedBadCase = computed(() => {
  return overview.value?.bad_cases.find((item) => item.id === selectedBadCaseId.value) || null
})

const metricMap = computed(() => {
  const metrics = overview.value?.latest_run.metrics || []
  return Object.fromEntries(metrics.map((metric) => [metric.name, metric.display_value]))
})

const baselineComparison = computed<BaselineComparison | null>(() => overview.value?.baseline_comparison || null)

const baselineCheckPassRate = computed<BaselineMetricComparison | null>(() => {
  return baselineComparison.value?.metric_comparisons.find((metric) => metric.name === 'check_pass_rate') || null
})

const regressedMetrics = computed<BaselineMetricComparison[]>(() => {
  return (baselineComparison.value?.metric_comparisons || []).filter((metric) => metric.regressed)
})

function formatRate(value: number | undefined): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '-'
}

function formatCount(value: number | undefined): string {
  if (typeof value !== 'number') {
    return '-'
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

function formatMetricValue(metric: BaselineMetricComparison): string {
  // 比率指标（*_rate，如 check_pass_rate / suite_pass_rate）按百分比显示；
  // 计数指标（如 failed_suites / failed_checks，direction=lower_is_better）显示原值。
  const isRateMetric = metric.name.endsWith('_rate')
  const baseline = isRateMetric ? formatRate(metric.baseline_value) : formatCount(metric.baseline_value)
  const candidate = isRateMetric ? formatRate(metric.candidate_value) : formatCount(metric.candidate_value)
  return `${metric.name}: ${baseline} → ${candidate}`
}

function splitList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

const sortedLayerCounts = computed(() => {
  const counts = overview.value?.bad_case_summary.layer_counts || {}
  return Object.entries(counts).sort((left, right) => right[1] - left[1])
})

const statusTypeMap: Record<string, 'success' | 'warning' | 'danger' | 'info'> = {
  open: 'danger',
  triaged: 'warning',
  fixed: 'success',
  regression_added: 'success',
  closed: 'info',
}

const severityTypeMap: Record<string, 'success' | 'warning' | 'danger' | 'info'> = {
  critical: 'danger',
  high: 'danger',
  medium: 'warning',
  low: 'info',
}

async function loadOverview() {
  loading.value = true
  try {
    const [evaluation, feedback] = await Promise.all([
      getEvaluationOverview(),
      getAiResponseFeedbackOverview(),
    ])
    overview.value = evaluation
    feedbackOverview.value = feedback
    selectedBadCaseId.value = overview.value.bad_cases[0]?.id || ''
    await nextTick()
    renderRegressionPie()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'AI 评估看板加载失败')
  } finally {
    loading.value = false
  }
}

function toTrendPoint(point: EvaluationHistoryPoint): [number, number] | null {
  // 历史快照早于 started_at 字段引入时后端输出 null，必须过滤这类点，否则时间轴错乱。
  if (!point.started_at) {
    return null
  }
  const timestamp = Date.parse(point.started_at)
  if (Number.isNaN(timestamp)) {
    return null
  }
  const value = point.check_pass_rate ?? point.pass_rate
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null
  }
  return [timestamp, value]
}

const trendAgentPoints = computed<[number, number][]>(() =>
  (trendHistory.value?.agent_eval || [])
    .map(toTrendPoint)
    .filter((point): point is [number, number] => point !== null)
    .sort((left, right) => left[0] - right[0]),
)

const trendRegressionPoints = computed<[number, number][]>(() =>
  (trendHistory.value?.production_regression || [])
    .map(toTrendPoint)
    .filter((point): point is [number, number] => point !== null)
    .sort((left, right) => left[0] - right[0]),
)

function toRagTrendPoint(point: EvaluationHistoryPoint): [number, number] | null {
  if (!point.started_at || point.hit_rate == null) {
    return null
  }
  const timestamp = Date.parse(point.started_at)
  if (Number.isNaN(timestamp)) {
    return null
  }
  return [timestamp, point.hit_rate]
}

const trendRagPoints = computed<[number, number][]>(() =>
  (trendHistory.value?.rag_retrieval || [])
    .map(toRagTrendPoint)
    .filter((point): point is [number, number] => point !== null)
    .sort((left, right) => left[0] - right[0]),
)

const trendEmpty = computed(
  () =>
    trendAgentPoints.value.length === 0 &&
    trendRegressionPoints.value.length === 0 &&
    trendRagPoints.value.length === 0,
)

function renderTrendChart() {
  if (!trendChartEl.value) {
    return
  }
  // 空态时图表 div 处于 display:none（v-show），此处 init 会触发
  // "Can't get DOM width or height" 警告并留下不可见实例；
  // 空态由 el-empty 展示，这里只需释放可能残留的旧实例。
  if (trendEmpty.value) {
    trendChart.value?.dispose()
    trendChart.value = null
    return
  }
  const chart = trendChart.value || echarts.init(trendChartEl.value)
  trendChart.value = chart
  chart.setOption({
    tooltip: {
      trigger: 'axis',
      formatter: (params: unknown) => {
        const items = (Array.isArray(params) ? params : [params]) as Array<{
          marker: string
          seriesName: string
          value: [number, number]
        }>
        return items
          .map((item) => {
            const [timestamp, value] = item.value
            const time = new Date(timestamp).toLocaleString('zh-CN', { hour12: false })
            const label = item.seriesName.includes('RAG') ? '命中率' : '通过率'
            return `${item.marker}${item.seriesName}（${time}）<br/>${label} ${(value * 100).toFixed(1)}%`
          })
          .join('<br/>')
      },
    },
    legend: { data: ['Agent 通过率', '回归通过率', 'RAG 检索命中率'] },
    grid: { left: 52, right: 20, top: 40, bottom: 32 },
    xAxis: { type: 'time' },
    yAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: {
        formatter: (value: unknown) => `${Math.round(Number(value) * 100)}%`,
      },
    },
    series: [
      {
        name: 'Agent 通过率',
        type: 'line',
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#409EFF', width: 2 },
        itemStyle: { color: '#409EFF' },
        data: trendAgentPoints.value,
      },
      {
        name: '回归通过率',
        type: 'line',
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#67C23A', width: 2 },
        itemStyle: { color: '#67C23A' },
        data: trendRegressionPoints.value,
      },
      {
        name: 'RAG 检索命中率',
        type: 'line',
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#E6A23C', width: 2 },
        itemStyle: { color: '#E6A23C' },
        data: trendRagPoints.value,
      },
    ],
  })
}

async function loadTrendHistory() {
  trendLoading.value = true
  try {
    trendHistory.value = await getEvaluationHistory()
    await nextTick()
    renderTrendChart()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : 'AI 评估历史加载失败')
  } finally {
    trendLoading.value = false
  }
}

function renderRegressionPie() {
  const run = overview.value?.latest_production_regression_run
  if (!run || !regressionPieEl.value) {
    return
  }
  const data = [
    { name: 'passed', value: run.passed_case_count },
    { name: 'failed', value: run.failed_case_count },
    { name: 'not_ready', value: run.not_ready_case_count },
    { name: 'error', value: run.error_case_count },
  ].filter((item) => item.value > 0)
  if (data.length === 0) {
    regressionPie.value?.dispose()
    regressionPie.value = null
    return
  }
  const chart = regressionPie.value || echarts.init(regressionPieEl.value)
  regressionPie.value = chart
  chart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c}（{d}%）' },
    legend: { bottom: 0 },
    color: ['#67C23A', '#F56C6C', '#E6A23C', '#909399'],
    series: [
      {
        type: 'pie',
        radius: ['42%', '68%'],
        center: ['50%', '44%'],
        avoidLabelOverlap: true,
        itemStyle: { borderRadius: 6, borderColor: '#fff', borderWidth: 2 },
        label: { show: true, formatter: '{b}: {c}' },
        data,
      },
    ],
  })
}

function onWindowResize() {
  trendChart.value?.resize()
  regressionPie.value?.resize()
  ragEvalChart.value?.resize()
}

function formatRagEvalTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

function renderRagEvalTrendChart() {
  if (!ragEvalChartEl.value) {
    return
  }
  if (ragEvalRuns.value.length === 0) {
    ragEvalChart.value?.dispose()
    ragEvalChart.value = null
    return
  }
  const chart = ragEvalChart.value || echarts.init(ragEvalChartEl.value)
  ragEvalChart.value = chart
  const runs = [...ragEvalRuns.value].reverse()
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0 },
    xAxis: {
      type: 'category',
      data: runs.map((run) => formatRagEvalTime(run.started_at)),
    },
    yAxis: { type: 'value', min: 0, max: 1 },
    series: [
      {
        name: 'keyword Hit@K',
        type: 'line',
        smooth: true,
        data: runs.map((run) => (run.retriever === 'keyword' ? run.hit_rate : null)),
      },
      {
        name: 'vector Hit@K',
        type: 'line',
        smooth: true,
        data: runs.map((run) => (run.retriever === 'vector' ? run.hit_rate : null)),
      },
    ],
  })
}

async function loadRagEvalRuns() {
  ragEvalLoading.value = true
  try {
    ragEvalRuns.value = await getRagEvalRuns(20)
    await nextTick()
    renderRagEvalTrendChart()
  } catch {
    ragEvalRuns.value = []
  } finally {
    ragEvalLoading.value = false
  }
}

function formatReportDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return new Date().toISOString().slice(0, 10).replace(/-/g, '')
  }
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}${month}${day}`
}

async function downloadReport(type: EvaluationReportType) {
  if (reportDownloading.value) {
    return
  }
  reportDownloading.value = true
  try {
    const report = await getLatestReport(type)
    const filename =
      type === 'agent'
        ? `agent-eval-report-${formatReportDate(report.generated_at)}.md`
        : `production-regression-report-${formatReportDate(report.generated_at)}.md`
    const blob = new Blob([report.report], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    try {
      link.click()
    } finally {
      link.remove()
    }
    // 延迟到下一个宏任务再释放，避免部分浏览器异步触发下载时被中断。
    setTimeout(() => URL.revokeObjectURL(url), 0)
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '报告下载失败')
  } finally {
    reportDownloading.value = false
  }
}

async function previewReportByType(type: EvaluationReportType) {
  try {
    const report = await getLatestReport(type)
    previewReport.value = report
    previewHtml.value = await marked.parse(report.report)
    previewVisible.value = true
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '报告预览失败')
  }
}

function handleReportCommand(command: string | number | object) {
  const key = command as string
  if (key.startsWith('preview-')) {
    void previewReportByType(key.replace('preview-', '') as EvaluationReportType)
  } else {
    void downloadReport(key as EvaluationReportType)
  }
}

function selectBadCase(item: BadCaseItem) {
  selectedBadCaseId.value = item.id
}

async function openFeedbackReview(candidate: AiFeedbackRegressionCandidate) {
  feedbackReviewVisible.value = true
  feedbackReviewLoading.value = true
  selectedFeedbackContext.value = null
  feedbackPromotionForm.failure_layer = 'agent_decision'
  feedbackPromotionForm.severity = 'medium'
  feedbackPromotionForm.failure_category = ''
  feedbackPromotionForm.expected_behavior = ''
  feedbackPromotionForm.recommended_action = ''
  feedbackPromotionForm.regression_action = ''
  feedbackPromotionForm.review_note = ''
  feedbackPromotionForm.regression_message = ''
  feedbackPromotionForm.regression_assertion = 'intent'
  feedbackPromotionForm.regression_expected_intent = null
  feedbackPromotionForm.regression_expected_tool = ''
  feedbackPromotionForm.regression_must_ask_fields = ''
  feedbackPromotionForm.regression_must_not_reveal_terms = ''
  try {
    selectedFeedbackContext.value = await getProductionFeedbackContext(candidate.feedback_id)
    feedbackPromotionForm.regression_message = selectedFeedbackContext.value.user_message_excerpt || ''
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '反馈上下文加载失败')
    feedbackReviewVisible.value = false
  } finally {
    feedbackReviewLoading.value = false
  }
}

watch(
  () => feedbackPromotionForm.regression_assertion,
  (assertion) => {
    if (assertion !== 'intent') {
      feedbackPromotionForm.regression_expected_intent = null
    }
    if (assertion !== 'tool_called') {
      feedbackPromotionForm.regression_expected_tool = ''
    }
    if (assertion !== 'must_ask_for') {
      feedbackPromotionForm.regression_must_ask_fields = ''
    }
    if (assertion !== 'must_not_reveal') {
      feedbackPromotionForm.regression_must_not_reveal_terms = ''
    }
  },
)

async function promoteSelectedFeedback() {
  const context = selectedFeedbackContext.value
  if (!context || feedbackPromotionSubmitting.value) {
    return
  }
  const requiredValues = [
    feedbackPromotionForm.failure_category,
    feedbackPromotionForm.expected_behavior,
    feedbackPromotionForm.recommended_action,
    feedbackPromotionForm.regression_action,
    feedbackPromotionForm.regression_message,
  ]
  const assertion = feedbackPromotionForm.regression_assertion
  const assertionConditionalInvalid =
    (assertion === 'intent' && !feedbackPromotionForm.regression_expected_intent) ||
    (assertion === 'tool_called' && !feedbackPromotionForm.regression_expected_tool.trim()) ||
    (assertion === 'must_ask_for' && splitList(feedbackPromotionForm.regression_must_ask_fields).length === 0) ||
    (assertion === 'must_not_reveal' && splitList(feedbackPromotionForm.regression_must_not_reveal_terms).length === 0)
  if (requiredValues.some((value) => !value.trim()) || assertionConditionalInvalid) {
    ElMessage.warning('请完成坏案例审核字段')
    return
  }
  feedbackPromotionSubmitting.value = true
  try {
    // 按当前断言只发送相关字段：非当前断言的字段以 undefined 省略，
    // 避免后端 Pydantic `str | None` 把显式 "" 解析为非 None 而 422。
    const payload: PromoteProductionFeedbackPayload = {
      ...feedbackPromotionForm,
      regression_expected_intent: assertion === 'intent' ? feedbackPromotionForm.regression_expected_intent : null,
      regression_expected_tool: assertion === 'tool_called' ? feedbackPromotionForm.regression_expected_tool : undefined,
      regression_must_ask_fields: assertion === 'must_ask_for' ? splitList(feedbackPromotionForm.regression_must_ask_fields) : undefined,
      regression_must_not_reveal_terms: assertion === 'must_not_reveal' ? splitList(feedbackPromotionForm.regression_must_not_reveal_terms) : undefined,
    }
    const response = await promoteProductionFeedback(context.feedback_id, payload)
    ElMessage.success(`已登记 ${response.bad_case.id}`)
    feedbackReviewVisible.value = false
    await loadOverview()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '坏案例登记失败')
  } finally {
    feedbackPromotionSubmitting.value = false
  }
}

async function runFormalProductionRegression() {
  if (productionRegressionRunning.value) {
    return
  }
  productionRegressionRunning.value = true
  try {
    const run = await runProductionRegression()
    if (overview.value) {
      overview.value.latest_production_regression_run = run
    }
    ElMessage.success(run.passed ? '正式 Bad Case 回归评测通过' : '正式 Bad Case 回归评测需要处理')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '正式 Bad Case 回归评测执行失败')
  } finally {
    productionRegressionRunning.value = false
  }
}

async function updateSelectedFeedbackReview(reviewStatus: 'triaged' | 'closed') {
  const context = selectedFeedbackContext.value
  if (!context || feedbackPromotionSubmitting.value) {
    return
  }
  feedbackPromotionSubmitting.value = true
  try {
    selectedFeedbackContext.value = await reviewProductionFeedback(context.feedback_id, {
      review_status: reviewStatus,
      review_note: feedbackPromotionForm.review_note,
    })
    ElMessage.success(reviewStatus === 'triaged' ? '审核状态已暂存' : '候选已关闭')
    await loadOverview()
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '审核状态更新失败')
  } finally {
    feedbackPromotionSubmitting.value = false
  }
}

onMounted(() => {
  void loadOverview()
  void loadTrendHistory()
  void loadRagEvalRuns()
  window.addEventListener('resize', onWindowResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onWindowResize)
  trendChart.value?.dispose()
  trendChart.value = null
  regressionPie.value?.dispose()
  regressionPie.value = null
})
</script>

<template>
  <section v-loading="loading" class="evaluation-page">
    <el-alert
      v-if="overview?.generated_from_latest_run"
      class="page-alert"
      type="info"
      :closable="false"
      title="当前 bad case 由本地评估结果即时生成；当 bad_cases.json 有正式记录后，页面会优先展示正式登记记录。"
    />

    <div class="page-grid">
      <el-card shadow="never" class="metric-card">
        <p>评估集</p>
        <div>
          <strong>{{ overview?.datasets.length || 0 }}</strong>
          <el-tag type="info">{{ overview?.registry_version || '-' }}</el-tag>
        </div>
      </el-card>
      <el-card shadow="never" class="metric-card">
        <p>线上负反馈</p>
        <div>
          <strong>{{ feedbackOverview?.unhelpful_count || 0 }}</strong>
          <el-tag type="warning">{{ ((feedbackOverview?.unhelpful_rate || 0) * 100).toFixed(1) }}%</el-tag>
        </div>
      </el-card>
      <el-card shadow="never" class="metric-card">
        <p>检查项</p>
        <div>
          <strong>{{ overview?.latest_run.evaluated_check_count || 0 }}</strong>
          <el-tag :type="overview?.latest_run.passed ? 'success' : 'danger'">
            {{ overview?.latest_run.passed ? '通过' : '有失败' }}
          </el-tag>
        </div>
      </el-card>
      <el-card shadow="never" class="metric-card">
        <p>通过率</p>
        <div>
          <strong>{{ metricMap.check_pass_rate || '-' }}</strong>
          <el-tag type="success">{{ overview?.latest_run.passed_check_count || 0 }} passed</el-tag>
        </div>
      </el-card>
      <el-card shadow="never" class="metric-card">
        <p>Bad Case</p>
        <div>
          <strong>{{ overview?.bad_case_summary.record_count || 0 }}</strong>
          <el-tag type="danger">{{ overview?.bad_case_summary.open_count || 0 }} open</el-tag>
        </div>
      </el-card>
    </div>

    <el-card v-if="overview" shadow="never" class="baseline-comparison-card">
      <template #header>
        <div class="card-header">
          <span>基线对比</span>
          <el-tag v-if="baselineComparison" :type="baselineComparison.regressed ? 'danger' : 'success'">
            {{ baselineComparison.regressed ? '有回归' : '无回归' }}
          </el-tag>
        </div>
      </template>
      <el-empty v-if="!baselineComparison" :image-size="52" description="暂无基线，运行一次后建立" />
      <template v-else>
        <el-descriptions :column="2" border>
          <el-descriptions-item label="数据集">{{ baselineComparison.dataset_name }}</el-descriptions-item>
          <el-descriptions-item label="数据集版本">{{ baselineComparison.dataset_version }}</el-descriptions-item>
          <el-descriptions-item label="基线 run_id">{{ baselineComparison.baseline_run_id }}</el-descriptions-item>
          <el-descriptions-item label="本次 run_id">{{ baselineComparison.candidate_run_id }}</el-descriptions-item>
          <el-descriptions-item label="基线版本">{{ baselineComparison.baseline_candidate_version }}</el-descriptions-item>
          <el-descriptions-item label="本次版本">{{ baselineComparison.candidate_version }}</el-descriptions-item>
        </el-descriptions>
        <div class="baseline-rate-grid">
          <div class="baseline-rate-item">
            <p>基线通过率</p>
            <strong>{{ baselineCheckPassRate ? formatRate(baselineCheckPassRate.baseline_value) : '-' }}</strong>
            <div class="baseline-rate-meta"><el-tag type="info" effect="plain">baseline</el-tag></div>
          </div>
          <div class="baseline-rate-item" :class="{ 'rate-regressed': baselineCheckPassRate?.regressed }">
            <p>本次通过率</p>
            <strong>{{ baselineCheckPassRate ? formatRate(baselineCheckPassRate.candidate_value) : '-' }}</strong>
            <div class="baseline-rate-meta">
              <el-tag :type="baselineCheckPassRate?.regressed ? 'danger' : 'success'" effect="plain">
                {{ baselineCheckPassRate ? (baselineCheckPassRate.regressed ? '较基线回归' : '未回归') : '-' }}
              </el-tag>
            </div>
          </div>
        </div>
        <div v-if="regressedMetrics.length" class="baseline-block">
          <h3>回归指标</h3>
          <div class="baseline-regressed-list">
            <el-tag v-for="metric in regressedMetrics" :key="metric.name" type="danger" effect="plain">
              {{ formatMetricValue(metric) }}
            </el-tag>
          </div>
        </div>
        <div v-if="baselineComparison.blocking_reasons.length" class="baseline-block">
          <h3>阻断原因</h3>
          <ul class="baseline-blocking-list">
            <li v-for="reason in baselineComparison.blocking_reasons" :key="reason">{{ reason }}</li>
          </ul>
        </div>
      </template>
    </el-card>

    <el-card shadow="never" class="baseline-comparison-card">
      <template #header>
        <div class="card-header">
          <span>评测趋势</span>
          <el-tag v-if="!trendEmpty" type="info" effect="plain">通过率</el-tag>
        </div>
      </template>
      <div v-loading="trendLoading" class="trend-chart-wrap">
        <el-empty v-if="trendEmpty" :image-size="52" description="暂无评估历史数据" />
        <div v-show="!trendEmpty" ref="trendChartEl" class="trend-chart"></div>
      </div>
    </el-card>

    <el-card shadow="never" class="baseline-comparison-card">
      <template #header>
        <div class="card-header">
          <span>RAG 检索评测历史</span>
          <el-tag v-if="ragEvalRuns.length" type="info" effect="plain">keyword / vector Hit@K</el-tag>
        </div>
      </template>
      <div v-loading="ragEvalLoading">
        <el-empty v-if="ragEvalRuns.length === 0" :image-size="52" description="暂无检索评测数据（运行 scripts/rag_retrieval_eval.py --save-run 后同步）" />
        <template v-else>
          <div ref="ragEvalChartEl" class="trend-chart"></div>
          <el-table class="rag-eval-table" :data="ragEvalRuns" height="220">
            <el-table-column label="时间" min-width="130">
              <template #default="{ row }">{{ formatRagEvalTime(row.started_at) }}</template>
            </el-table-column>
            <el-table-column prop="retriever" label="模式" width="90">
              <template #default="{ row }">
                <el-tag :type="row.retriever === 'vector' ? 'primary' : 'info'">{{ row.retriever }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="case_count" label="用例数" width="80" />
            <el-table-column label="Hit@K" width="90">
              <template #default="{ row }">{{ (row.hit_rate * 100).toFixed(1) }}%</template>
            </el-table-column>
            <el-table-column label="Recall@K" width="100">
              <template #default="{ row }">{{ (row.recall * 100).toFixed(1) }}%</template>
            </el-table-column>
            <el-table-column label="Precision@K" width="110">
              <template #default="{ row }">{{ (row.precision * 100).toFixed(1) }}%</template>
            </el-table-column>
            <el-table-column label="MRR" width="90">
              <template #default="{ row }">{{ (row.mrr * 100).toFixed(1) }}%</template>
            </el-table-column>
          </el-table>
        </template>
      </div>
    </el-card>

    <section class="content-grid two-columns">
      <div class="content-grid">
        <el-card shadow="never">
          <template #header>线上反馈与回归候选</template>
          <el-descriptions :column="3" border>
            <el-descriptions-item label="总反馈">{{ feedbackOverview?.total_count || 0 }}</el-descriptions-item>
            <el-descriptions-item label="有帮助">{{ feedbackOverview?.helpful_count || 0 }}</el-descriptions-item>
            <el-descriptions-item label="无帮助">{{ feedbackOverview?.unhelpful_count || 0 }}</el-descriptions-item>
          </el-descriptions>
          <el-table class="feedback-candidate-table" :data="feedbackOverview?.regression_candidates || []" height="240">
            <el-table-column prop="reason" label="原因" width="150" />
            <el-table-column prop="agent_route" label="链路" width="150" />
            <el-table-column prop="citation_count" label="引用数" width="90" />
            <el-table-column label="人工转接" width="110">
              <template #default="{ row }">
                <el-tag :type="row.human_handoff_suggested ? 'warning' : 'info'">
                  {{ row.human_handoff_suggested ? '建议过' : '未建议' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="review_status" label="状态" width="130" />
            <el-table-column prop="trace_id" label="trace_id" min-width="180" show-overflow-tooltip />
            <el-table-column label="操作" width="90" fixed="right">
              <template #default="{ row }">
                <el-button link type="primary" @click="openFeedbackReview(row)">
                  {{ row.review_status === 'regression_added' ? '查看' : '审核' }}
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-empty
            v-if="feedbackOverview && feedbackOverview.regression_candidates.length === 0"
            :image-size="52"
            description="暂无线上负反馈候选"
          />
        </el-card>

        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>本地评估运行</span>
              <div class="card-actions">
                <el-dropdown trigger="click" @command="handleReportCommand">
                  <el-button :loading="reportDownloading" plain>
                    报告<el-icon class="el-icon--right"><ArrowDown /></el-icon>
                  </el-button>
                  <template #dropdown>
                    <el-dropdown-menu>
                      <el-dropdown-item divided command="preview-agent">预览 Agent 评测报告</el-dropdown-item>
                      <el-dropdown-item command="preview-regression">预览生产回归报告</el-dropdown-item>
                      <el-dropdown-item divided command="agent">下载 Agent 评测报告</el-dropdown-item>
                      <el-dropdown-item command="regression">下载生产回归报告</el-dropdown-item>
                    </el-dropdown-menu>
                  </template>
                </el-dropdown>
                <el-button type="primary" plain :loading="loading" @click="loadOverview">刷新</el-button>
              </div>
            </div>
          </template>
          <el-descriptions v-if="overview" :column="2" border>
            <el-descriptions-item label="run_id">{{ overview.latest_run.run_id }}</el-descriptions-item>
            <el-descriptions-item label="数据集">
              {{ overview.latest_run.dataset_name }}:{{ overview.latest_run.dataset_version }}
            </el-descriptions-item>
            <el-descriptions-item label="候选版本">{{ overview.latest_run.candidate_version }}</el-descriptions-item>
            <el-descriptions-item label="模型">{{ overview.latest_run.model_name }}</el-descriptions-item>
            <el-descriptions-item label="样例数">{{ overview.latest_run.selected_case_count }}</el-descriptions-item>
            <el-descriptions-item label="trace_id">{{ overview.trace_id }}</el-descriptions-item>
          </el-descriptions>
        </el-card>

        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>正式 Bad Case 回归评测</span>
              <el-button type="primary" :loading="productionRegressionRunning" @click="runFormalProductionRegression">
                运行回归
              </el-button>
            </div>
          </template>
          <el-empty
            v-if="!overview?.latest_production_regression_run"
            :image-size="52"
            description="尚未运行正式 Bad Case 回归评测"
          />
          <template v-else>
            <el-descriptions :column="3" border>
              <el-descriptions-item label="运行 ID">{{ overview.latest_production_regression_run.run_id }}</el-descriptions-item>
              <el-descriptions-item label="通过">{{ overview.latest_production_regression_run.passed_case_count }}</el-descriptions-item>
              <el-descriptions-item label="失败">{{ overview.latest_production_regression_run.failed_case_count }}</el-descriptions-item>
              <el-descriptions-item label="待补充">{{ overview.latest_production_regression_run.not_ready_case_count }}</el-descriptions-item>
              <el-descriptions-item label="执行异常">{{ overview.latest_production_regression_run.error_case_count }}</el-descriptions-item>
              <el-descriptions-item label="结论">
                <el-tag :type="overview.latest_production_regression_run.passed ? 'success' : 'warning'">
                  {{ overview.latest_production_regression_run.passed ? '通过' : '需要处理' }}
                </el-tag>
              </el-descriptions-item>
            </el-descriptions>
            <div v-if="overview?.latest_production_regression_run" ref="regressionPieEl" class="regression-pie"></div>
            <el-table :data="overview.latest_production_regression_run.results" height="260" class="feedback-candidate-table">
              <el-table-column prop="title" label="Bad Case" min-width="180" show-overflow-tooltip />
              <el-table-column prop="assertion" label="断言" width="180" />
              <el-table-column prop="expected" label="期望" width="130" />
              <el-table-column prop="actual" label="实际" width="130" />
              <el-table-column label="结果" width="110">
                <template #default="{ row }">
                  <el-tag :type="row.outcome === 'passed' ? 'success' : row.outcome === 'failed' ? 'danger' : 'warning'">
                    {{ row.outcome }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="detail" label="说明" min-width="220" show-overflow-tooltip />
            </el-table>
          </template>
        </el-card>

        <el-card shadow="never">
          <template #header>评估套件</template>
          <el-table :data="overview?.latest_run.suites || []" height="260">
            <el-table-column prop="name" label="套件" width="110" />
            <el-table-column prop="title" label="说明" min-width="210" show-overflow-tooltip />
            <el-table-column prop="case_count" label="样例" width="80" />
            <el-table-column prop="failed_case_count" label="失败" width="80" />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="row.passed ? 'success' : 'danger'">
                  {{ row.passed ? '通过' : '失败' }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never">
          <template #header>评估集登记</template>
          <el-table :data="overview?.datasets || []" height="260">
            <el-table-column prop="name" label="名称" width="170" />
            <el-table-column prop="version" label="版本" width="120" />
            <el-table-column prop="task_type" label="类型" width="130" />
            <el-table-column prop="description" label="说明" min-width="260" show-overflow-tooltip />
            <el-table-column label="冻结" width="80">
              <template #default="{ row }">
                <el-tag :type="row.frozen ? 'success' : 'warning'">{{ row.frozen ? '是' : '否' }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <div class="content-grid">
        <el-card shadow="never">
          <template #header>失败层分布</template>
          <div v-if="sortedLayerCounts.length" class="layer-list">
            <div v-for="[layer, count] in sortedLayerCounts" :key="layer" class="layer-row">
              <span>{{ layer }}</span>
              <el-progress :percentage="Math.min(100, count * 20)" :format="() => `${count}`" />
            </div>
          </div>
          <el-empty v-else description="暂无失败层统计" />
        </el-card>

        <el-card shadow="never">
          <template #header>Bad Case 列表</template>
          <el-table
            :data="overview?.bad_cases || []"
            height="300"
            highlight-current-row
            row-key="id"
            @row-click="selectBadCase"
          >
            <el-table-column prop="failure_layer" label="层" width="120" />
            <el-table-column prop="title" label="标题" min-width="180" show-overflow-tooltip />
            <el-table-column label="级别" width="86">
              <template #default="{ row }">
                <el-tag :type="severityTypeMap[row.severity] || 'info'">{{ row.severity }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="statusTypeMap[row.status] || 'info'">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
          <el-empty v-if="overview && overview.bad_cases.length === 0" description="暂无 bad case" />
        </el-card>

        <el-card shadow="never">
          <template #header>Bad Case 详情</template>
          <el-empty v-if="!selectedBadCase" description="请选择一个 bad case" />
          <div v-else class="bad-case-detail">
            <h2>{{ selectedBadCase.title }}</h2>
            <dl class="detail-list">
              <div>
                <dt>ID</dt>
                <dd>{{ selectedBadCase.id }}</dd>
              </div>
              <div>
                <dt>失败层</dt>
                <dd>{{ selectedBadCase.failure_layer }}</dd>
              </div>
              <div>
                <dt>分类</dt>
                <dd>{{ selectedBadCase.failure_category }}</dd>
              </div>
              <div>
                <dt>任务</dt>
                <dd>{{ selectedBadCase.task_type }}</dd>
              </div>
            </dl>
            <section>
              <h3>期望行为</h3>
              <p>{{ selectedBadCase.expected_behavior }}</p>
            </section>
            <section>
              <h3>实际问题</h3>
              <p>{{ selectedBadCase.actual_behavior }}</p>
            </section>
            <section>
              <h3>建议动作</h3>
              <p>{{ selectedBadCase.recommended_action }}</p>
            </section>
            <section>
              <h3>回归策略</h3>
              <p>{{ selectedBadCase.regression_action }}</p>
            </section>
            <section>
              <h3>证据摘要</h3>
              <p>{{ selectedBadCase.evidence_summary }}</p>
            </section>
          </div>
        </el-card>
      </div>
    </section>

    <el-dialog v-model="feedbackReviewVisible" title="线上反馈审核" width="min(860px, 92vw)" destroy-on-close>
      <div v-loading="feedbackReviewLoading" class="feedback-review-dialog">
        <template v-if="selectedFeedbackContext">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="反馈原因">{{ selectedFeedbackContext.reason || '未填写' }}</el-descriptions-item>
            <el-descriptions-item label="Agent 路由">{{ selectedFeedbackContext.agent_route }}</el-descriptions-item>
            <el-descriptions-item label="trace_id" :span="2">{{ selectedFeedbackContext.trace_id }}</el-descriptions-item>
          </el-descriptions>
          <section class="feedback-context-block">
            <h3>用户问题</h3>
            <p>{{ selectedFeedbackContext.user_message_excerpt || '历史问题已过期或不可用' }}</p>
          </section>
          <section class="feedback-context-block">
            <h3>AI 最终回答</h3>
            <p>{{ selectedFeedbackContext.assistant_answer_excerpt || '历史回答已过期或不可用' }}</p>
          </section>
          <section v-if="selectedFeedbackContext.citation_summary.length" class="feedback-context-block">
            <h3>引用摘要</h3>
            <p v-for="citation in selectedFeedbackContext.citation_summary" :key="citation.chunk_id || citation.source || citation.title || 'citation'">
              {{ citation.title || citation.source }}
            </p>
          </section>
          <el-form class="feedback-review-form" label-position="top">
            <el-form-item label="失败层">
              <el-select v-model="feedbackPromotionForm.failure_layer">
                <el-option label="意图识别" value="intent" />
                <el-option label="路由" value="routing" />
                <el-option label="RAG 检索" value="rag_retrieval" />
                <el-option label="RAG 引用" value="rag_citation" />
                <el-option label="Agent 决策" value="agent_decision" />
                <el-option label="工具调用" value="tool_calling" />
                <el-option label="权限" value="permission" />
                <el-option label="安全" value="security" />
                <el-option label="模型输出" value="model_output" />
              </el-select>
            </el-form-item>
            <el-form-item label="严重程度">
              <el-select v-model="feedbackPromotionForm.severity">
                <el-option label="低" value="low" />
                <el-option label="中" value="medium" />
                <el-option label="高" value="high" />
                <el-option label="严重" value="critical" />
              </el-select>
            </el-form-item>
            <el-form-item label="失败分类"><el-input v-model="feedbackPromotionForm.failure_category" maxlength="120" /></el-form-item>
            <el-form-item label="预期行为"><el-input v-model="feedbackPromotionForm.expected_behavior" type="textarea" :rows="2" maxlength="1000" /></el-form-item>
            <el-form-item label="建议改动"><el-input v-model="feedbackPromotionForm.recommended_action" type="textarea" :rows="2" maxlength="1000" /></el-form-item>
            <el-form-item label="回归策略"><el-input v-model="feedbackPromotionForm.regression_action" type="textarea" :rows="2" maxlength="1000" /></el-form-item>
            <el-form-item label="审核备注"><el-input v-model="feedbackPromotionForm.review_note" type="textarea" :rows="2" maxlength="1000" /></el-form-item>
            <el-form-item label="回归输入">
              <el-input v-model="feedbackPromotionForm.regression_message" type="textarea" :rows="2" maxlength="4000" />
            </el-form-item>
            <el-form-item label="自动断言">
              <el-select v-model="feedbackPromotionForm.regression_assertion">
                <el-option label="意图必须正确" value="intent" />
                <el-option label="必须返回引用" value="citation_present" />
                <el-option label="必须进入工单确认" value="ticket_confirmation_required" />
                <el-option label="必须调用指定工具" value="tool_called" />
                <el-option label="必须追问指定字段" value="must_ask_for" />
                <el-option label="不得泄露指定内容" value="must_not_reveal" />
              </el-select>
            </el-form-item>
            <el-form-item v-if="feedbackPromotionForm.regression_assertion === 'intent'" label="期望意图">
              <el-select v-model="feedbackPromotionForm.regression_expected_intent">
                <el-option label="政策问答" value="policy_question" />
                <el-option label="订单查询" value="order_query" />
                <el-option label="工单请求" value="ticket_request" />
                <el-option label="退款申请" value="refund_request" />
                <el-option label="取消申请" value="cancel_request" />
                <el-option label="闲聊" value="smalltalk" />
                <el-option label="不支持" value="unsupported" />
                <el-option label="不明确" value="unclear" />
              </el-select>
            </el-form-item>
            <el-form-item v-if="feedbackPromotionForm.regression_assertion === 'tool_called'" label="期望工具">
              <el-input v-model="feedbackPromotionForm.regression_expected_tool" maxlength="200" placeholder="如：query_order / refund_order / create_ticket（工具名或执行节点名均可）" />
            </el-form-item>
            <el-form-item v-if="feedbackPromotionForm.regression_assertion === 'must_ask_for'" label="必须追问字段">
              <el-input
                v-model="feedbackPromotionForm.regression_must_ask_fields"
                maxlength="500"
                placeholder="逗号分隔，如：订单号, 联系电话"
              />
            </el-form-item>
            <el-form-item v-if="feedbackPromotionForm.regression_assertion === 'must_not_reveal'" label="不得泄露内容">
              <el-input
                v-model="feedbackPromotionForm.regression_must_not_reveal_terms"
                maxlength="500"
                placeholder="逗号分隔，如：优惠券码, 退款金额"
              />
            </el-form-item>
          </el-form>
        </template>
      </div>
      <template #footer>
        <el-button @click="feedbackReviewVisible = false">取消</el-button>
        <el-button :disabled="feedbackPromotionSubmitting" @click="updateSelectedFeedbackReview('closed')">关闭候选</el-button>
        <el-button :disabled="feedbackPromotionSubmitting" @click="updateSelectedFeedbackReview('triaged')">暂存审核</el-button>
        <el-button type="primary" :loading="feedbackPromotionSubmitting" @click="promoteSelectedFeedback">登记为 Bad Case</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="previewVisible"
      :title="previewReport ? (previewReport.type === 'agent' ? 'Agent 评测报告' : '生产回归报告') : '评测报告'"
      width="800px"
      top="5vh"
      destroy-on-close
    >
      <div v-if="previewReport" class="preview-meta">
        生成时间：{{ new Date(previewReport.generated_at).toLocaleString('zh-CN', { hour12: false }) }}
      </div>
      <div class="preview-markdown" v-html="previewHtml"></div>
      <template #footer>
        <el-button @click="previewVisible = false">关闭</el-button>
      </template>
    </el-dialog>
  </section>
</template>

<style scoped>
.preview-meta {
  margin-bottom: 12px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.preview-markdown {
  max-height: 70vh;
  overflow: auto;
  padding: 4px 8px;
  line-height: 1.7;
}
.preview-markdown pre {
  background: #f6f8fa;
  padding: 10px;
  border-radius: 6px;
  overflow: auto;
}
.preview-markdown table {
  border-collapse: collapse;
}
.preview-markdown th,
.preview-markdown td {
  border: 1px solid var(--el-border-color);
  padding: 4px 8px;
}
.regression-pie {
  height: 180px;
  margin: 8px 0 4px;
}
.evaluation-page {
  min-height: 520px;
}

.layer-list {
  display: grid;
  gap: 12px;
}

.layer-row {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
}

.layer-row span {
  color: #344054;
  font-size: 13px;
  overflow-wrap: anywhere;
}

.bad-case-detail h2 {
  margin: 0 0 16px;
  color: #101828;
  font-size: 18px;
  line-height: 1.4;
}

.bad-case-detail section {
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
  margin-top: 12px;
}

.bad-case-detail h3 {
  margin: 0 0 6px;
  color: #344054;
  font-size: 14px;
}

.bad-case-detail p {
  margin: 0;
  color: #101828;
  line-height: 1.7;
  overflow-wrap: anywhere;
}

.feedback-candidate-table {
  margin-top: 16px;
}

.baseline-comparison-card {
  margin-bottom: 16px;
}

.card-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.trend-chart-wrap {
  min-height: 300px;
}

.trend-chart {
  width: 100%;
  height: 300px;
}

.baseline-rate-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.baseline-rate-item {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px 16px;
}

.baseline-rate-item p {
  margin: 0 0 8px;
  color: #667085;
  font-size: 13px;
}

.baseline-rate-item strong {
  font-size: 24px;
  color: #101828;
  line-height: 30px;
}

.baseline-rate-item.rate-regressed strong {
  color: #f56c6c;
}

.baseline-rate-meta {
  margin-top: 8px;
}

.baseline-block {
  margin-top: 16px;
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
}

.baseline-block h3 {
  margin: 0 0 8px;
  font-size: 14px;
  color: #344054;
}

.baseline-regressed-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.baseline-blocking-list {
  margin: 0;
  padding-left: 18px;
  color: #f56c6c;
  line-height: 1.8;
}

.feedback-review-dialog {
  display: grid;
  gap: 16px;
}

.feedback-context-block {
  border-top: 1px solid var(--el-border-color);
  padding-top: 12px;
}

.feedback-context-block h3 {
  margin: 0 0 6px;
  font-size: 14px;
}

.feedback-context-block p {
  margin: 4px 0;
  line-height: 1.65;
  white-space: pre-wrap;
}

.feedback-review-form {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 12px;
}

.feedback-review-form :deep(.el-form-item:nth-child(n + 3)) {
  grid-column: 1 / -1;
}

@media (max-width: 900px) {
  .layer-row {
    grid-template-columns: 1fr;
  }
  .feedback-review-form { grid-template-columns: 1fr; }
}
</style>
