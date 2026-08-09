import { aiApi } from './http'

export interface EvaluationDataset {
  name: string
  version: string
  task_type: string
  description: string
  frozen: boolean
  baseline_run_id: string | null
  tags: string[]
}

export interface EvaluationMetric {
  name: string
  value: number
  direction: 'higher_is_better' | 'lower_is_better'
  display_value: string
}

export interface EvaluationSuite {
  name: string
  title: string
  case_count: number
  failed_case_count: number
  passed: boolean
}

export interface EvaluationRunOverview {
  run_id: string
  dataset_name: string
  dataset_version: string
  candidate_version: string
  model_name: string
  selected_case_count: number
  evaluated_check_count: number
  passed_check_count: number
  failed_check_count: number
  passed: boolean
  metrics: EvaluationMetric[]
  suites: EvaluationSuite[]
}

export interface ProductionRegressionCaseResult {
  bad_case_id: string
  title: string
  outcome: 'passed' | 'failed' | 'not_ready' | 'error'
  assertion: string | null
  expected: string | null
  actual: string | null
  detail: string
}

export interface ProductionRegressionRun {
  run_id: string
  started_at: string
  completed_at: string
  total_case_count: number
  passed_case_count: number
  failed_case_count: number
  not_ready_case_count: number
  error_case_count: number
  passed: boolean
  results: ProductionRegressionCaseResult[]
}

export interface BadCaseSummary {
  record_count: number
  open_count: number
  regression_added_count: number
  severity_counts: Record<string, number>
  status_counts: Record<string, number>
  layer_counts: Record<string, number>
}

export interface BadCaseItem {
  id: string
  title: string
  source: string
  task_type: string
  severity: string
  status: string
  failure_layer: string
  failure_category: string
  expected_behavior: string
  actual_behavior: string
  root_cause: string
  recommended_action: string
  regression_action: string
  evidence_summary: string
  tags: string[]
}

export interface BaselineMetricComparison {
  name: string
  baseline_value: number
  candidate_value: number
  delta: number
  direction: 'higher_is_better' | 'lower_is_better'
  regressed: boolean
}

export interface BaselineComparison {
  dataset_name: string
  dataset_version: string
  baseline_run_id: string
  candidate_run_id: string
  baseline_candidate_version: string
  candidate_version: string
  regressed: boolean
  blocking_reasons: string[]
  metric_comparisons: BaselineMetricComparison[]
}

export interface EvaluationOverview {
  registry_version: string
  datasets: EvaluationDataset[]
  latest_run: EvaluationRunOverview
  bad_case_summary: BadCaseSummary
  bad_cases: BadCaseItem[]
  generated_from_latest_run: boolean
  latest_production_regression_run: ProductionRegressionRun | null
  baseline_comparison: BaselineComparison | null
  trace_id: string
}

/**
 * 单次评估快照的通过率点。
 * 注意：历史快照早于 started_at 字段引入时，后端输出 started_at 为 null，
 * 前端趋势图必须过滤这类点，否则时间轴错乱。
 */
export interface EvaluationHistoryPoint {
  started_at: string | null
  check_pass_rate: number | null
  passed: number | null
  total: number | null
  pass_rate: number | null
  hit_rate: number | null
  recall: number | null
  mrr: number | null
}

export interface EvaluationHistory {
  agent_eval: EvaluationHistoryPoint[]
  production_regression: EvaluationHistoryPoint[]
  rag_retrieval: EvaluationHistoryPoint[]
}

export type EvaluationReportType = 'agent' | 'regression'

export interface EvaluationReport {
  report: string
  type: string
  generated_at: string
}

interface AiServiceErrorBody {
  code?: string
  message?: string
  trace_id?: string
}

export async function getEvaluationOverview() {
  try {
    const response = await aiApi.get<EvaluationOverview>('/api/ai/evaluation/overview')
    return response.data
  } catch (error: any) {
    const data = error?.response?.data as AiServiceErrorBody | undefined
    throw new Error(data?.message || 'AI 评估看板加载失败')
  }
}

export async function runProductionRegression() {
  try {
    const response = await aiApi.post<ProductionRegressionRun>('/api/ai/evaluation/runs/production-regression')
    return response.data
  } catch (error: any) {
    const data = error?.response?.data as AiServiceErrorBody | undefined
    throw new Error(data?.message || 'Production regression run failed')
  }
}

export async function getEvaluationHistory() {
  try {
    const response = await aiApi.get<EvaluationHistory>('/api/ai/evaluation/history')
    return response.data
  } catch (error: any) {
    const data = error?.response?.data as AiServiceErrorBody | undefined
    throw new Error(data?.message || 'AI 评估历史加载失败')
  }
}

export async function getLatestReport(type: EvaluationReportType) {
  try {
    const response = await aiApi.get<EvaluationReport>('/api/ai/evaluation/reports/latest', {
      params: { type },
    })
    return response.data
  } catch (error: any) {
    if (error?.response?.status === 404) {
      throw new Error('暂无报告数据')
    }
    const data = error?.response?.data as AiServiceErrorBody | undefined
    throw new Error(data?.message || 'AI 评估报告加载失败')
  }
}

export interface RagEvalRun {
  run_id: string
  retriever: 'keyword' | 'vector'
  started_at: string
  top_k: number
  hit_rate: number
  recall: number
  precision: number
  mrr: number
  case_count: number
}

export async function getRagEvalRuns(limit = 20): Promise<RagEvalRun[]> {
  try {
    const response = await aiApi.get<{ runs: RagEvalRun[] }>('/api/ai/evaluation/rag-retrieval-runs', {
      params: { limit },
    })
    return response.data?.runs ?? []
  } catch {
    return []
  }
}
