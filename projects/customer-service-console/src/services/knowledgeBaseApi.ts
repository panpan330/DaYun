import { aiApi } from './http'

export type KnowledgeBaseEmbeddingMode = 'fake' | 'real'

export interface KnowledgeBaseDocumentStatus {
  source: string
  title: string
  file_name: string
  file_extension: string
  doc_type: string | null
  business_domain: string | null
  permission_group: string | null
}

export interface KnowledgeBaseStatusResponse {
  documents: KnowledgeBaseDocumentStatus[]
  document_count: number
  collection_name: string
  qdrant_base_url: string
  fake_embedding_dimension: number
  real_embedding_configured: boolean
  trace_id: string
}

export interface KnowledgeBaseIngestRequest {
  embedding_mode: KnowledgeBaseEmbeddingMode
  refresh: boolean
  wait: boolean
  include_readme: boolean
  chunk_size: number
  chunk_overlap: number
}

export interface KnowledgeBaseIngestResponse {
  embedding_mode: KnowledgeBaseEmbeddingMode
  document_count: number
  chunk_count: number
  vector_count: number
  vector_dimension: number
  collection_name: string
  replaced_source_count: number
  trace_id: string
}

interface AiServiceErrorBody {
  code?: string
  message?: string
  trace_id?: string
}

export async function getKnowledgeBaseStatus() {
  try {
    const response = await aiApi.get<KnowledgeBaseStatusResponse>('/api/knowledge-base/status')
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识库状态加载失败')
  }
}

export async function ingestKnowledgeBase(request: KnowledgeBaseIngestRequest) {
  try {
    const response = await aiApi.post<KnowledgeBaseIngestResponse>('/api/knowledge-base/ingest', request)
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识库入库失败')
  }
}

function buildAiServiceError(error: any, fallbackMessage: string) {
  const data = error?.response?.data as AiServiceErrorBody | undefined
  if (data?.message) {
    return new Error(data.message)
  }
  return new Error(fallbackMessage)
}

export interface KnowledgeBaseCollectionStatus {
  collection_name: string
  knowledge_base_ids: string[]
  display_name: string
  point_count: number
  exists: boolean
  is_legacy: boolean
}

export interface KnowledgeBaseCollectionsResponse {
  collections: KnowledgeBaseCollectionStatus[]
  legacy_collections: KnowledgeBaseCollectionStatus[]
  trace_id: string
}

export interface KnowledgeBaseDocumentItem {
  document_id: string
  title: string
  business_domain: string
  permission_group: string
  doc_type: string
  collection_name: string
  chunk_count: number
  source_file_name: string
  exists_local: boolean
  status: string
  updated_at: string | null
}

export interface KnowledgeBaseDocumentCreateRequest {
  document_id: string
  title: string
  content: string
  business_domain: string
  permission_group: string
  doc_type: string
  collection_name: string
  embedding_mode: KnowledgeBaseEmbeddingMode
  chunk_size: number
  chunk_overlap: number
}

export interface KnowledgeBaseDocumentListResponse {
  documents: KnowledgeBaseDocumentItem[]
  document_count: number
  trace_id: string
}

export async function getKnowledgeBaseCollections() {
  try {
    const response = await aiApi.get<KnowledgeBaseCollectionsResponse>('/api/knowledge-base/collections')
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识库 collection 加载失败')
  }
}

export async function listKnowledgeBaseDocuments() {
  try {
    const response = await aiApi.get<KnowledgeBaseDocumentListResponse>('/api/knowledge-base/documents')
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识库文档列表加载失败')
  }
}

export async function createKnowledgeBaseDocument(request: KnowledgeBaseDocumentCreateRequest) {
  try {
    const response = await aiApi.post<KnowledgeBaseDocumentItem>('/api/knowledge-base/documents', request)
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识文档上传失败')
  }
}

export async function uploadKnowledgeBaseDocument(formData: FormData) {
  try {
    const response = await aiApi.post<KnowledgeBaseDocumentItem>(
      '/api/knowledge-base/documents/upload',
      formData,
    )
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识文档上传失败')
  }
}

export async function updateKnowledgeBaseDocument(documentId: string, request: Partial<KnowledgeBaseDocumentCreateRequest>) {
  try {
    const response = await aiApi.put<KnowledgeBaseDocumentItem>(`/api/knowledge-base/documents/${documentId}`, request)
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识文档更新失败')
  }
}

export async function deleteKnowledgeBaseDocument(documentId: string) {
  try {
    const response = await aiApi.delete<{ success: boolean }>(`/api/knowledge-base/documents/${documentId}`)
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识文档删除失败')
  }
}

export async function ingestKnowledgeBaseDocument(
  documentId: string,
  request: { embedding_mode: KnowledgeBaseEmbeddingMode; chunk_size: number; chunk_overlap: number },
) {
  try {
    const response = await aiApi.post<KnowledgeBaseDocumentItem>(`/api/knowledge-base/documents/${documentId}/ingest`, request)
    return response.data
  } catch (error: any) {
    throw buildAiServiceError(error, '知识文档同步失败')
  }
}
