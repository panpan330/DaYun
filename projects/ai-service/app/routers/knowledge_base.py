import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.exceptions import AppException
from app.core.trace import get_trace_id
from app.rag.embeddings import (
    DeterministicHashEmbeddingModel,
    EmbeddingModel,
    OpenAICompatibleEmbeddingModel,
)
from app.rag.ingestion import (
    VectorStoreUpdater,
    delete_document_from_vector_store,
    ingest_directory_to_vector_store,
    refresh_directory_in_vector_store,
    update_single_document,
)
from app.rag.knowledge_routing import default_rag_knowledge_bases
from app.rag.loaders import (
    METADATA_KEY_MAP,
    load_document,
    load_documents_from_directory,
)
from app.rag.vector_store import QdrantVectorStore
from app.schemas.knowledge_base import (
    KnowledgeBaseCollectionStatus,
    KnowledgeBaseCollectionsResponse,
    KnowledgeBaseDocumentCreateRequest,
    KnowledgeBaseDocumentIngestRequest,
    KnowledgeBaseDocumentListView,
    KnowledgeBaseDocumentStatus,
    KnowledgeBaseDocumentUpdateRequest,
    KnowledgeBaseDocumentView,
    KnowledgeBaseIngestRequest,
    KnowledgeBaseIngestResponse,
    KnowledgeBaseStatusResponse,
)


router = APIRouter(prefix="/api/knowledge-base", tags=["knowledge-base"])
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def get_knowledge_base_dir() -> Path:
    return KNOWLEDGE_BASE_DIR


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreUpdater:
    return QdrantVectorStore.from_settings(settings)


def build_collection_vector_store(
    settings: Settings,
    *,
    collection_name: str | None = None,
) -> QdrantVectorStore:
    return QdrantVectorStore.from_settings(
        settings, collection_name=collection_name
    )


def build_embedding_model(
    request: KnowledgeBaseIngestRequest,
    settings: Settings,
) -> EmbeddingModel:
    if request.embedding_mode == "fake":
        return DeterministicHashEmbeddingModel(dimension=settings.qdrant_vector_size)
    try:
        return OpenAICompatibleEmbeddingModel.from_settings(settings)
    except ValueError as exc:
        raise AppException(
            code="EMBEDDING_API_KEY_MISSING",
            message="Embedding API key 未配置，无法执行真实 embedding 入库。",
            status_code=500,
        ) from exc


@router.get("/collections", response_model=KnowledgeBaseCollectionsResponse)
def knowledge_base_collections(
    settings: Settings = Depends(get_settings),
) -> KnowledgeBaseCollectionsResponse:
    definitions = default_rag_knowledge_bases()
    managed: dict[str, list] = {}
    for definition in definitions:
        managed.setdefault(definition.collection_name, []).append(definition)

    legacy_name = settings.qdrant_collection_name
    all_names = set(build_collection_vector_store(settings).list_collections())

    collections: list[KnowledgeBaseCollectionStatus] = []
    for collection_name in sorted(managed):
        defs = managed[collection_name]
        exists = collection_name in all_names
        point_count = 0
        if exists:
            point_count = build_collection_vector_store(
                settings, collection_name=collection_name
            ).count_points()
        collections.append(
            KnowledgeBaseCollectionStatus(
                collection_name=collection_name,
                knowledge_base_ids=[d.knowledge_base_id for d in defs],
                display_name=" / ".join(d.display_name for d in defs),
                point_count=point_count,
                exists=exists,
                is_legacy=False,
            )
        )

    legacy: list[KnowledgeBaseCollectionStatus] = []
    if legacy_name and legacy_name not in managed:
        legacy_exists = legacy_name in all_names
        legacy.append(
            KnowledgeBaseCollectionStatus(
                collection_name=legacy_name,
                knowledge_base_ids=[],
                display_name="Legacy single collection",
                point_count=(
                    build_collection_vector_store(settings).count_points()
                    if legacy_exists
                    else 0
                ),
                exists=legacy_exists,
                is_legacy=True,
            )
        )

    return KnowledgeBaseCollectionsResponse(
        collections=collections,
        legacy_collections=legacy,
        trace_id=get_trace_id(),
    )


@router.get("/status", response_model=KnowledgeBaseStatusResponse)
def knowledge_base_status(
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseStatusResponse:
    try:
        documents = load_documents_from_directory(directory)
    except NotADirectoryError as exc:
        raise AppException(
            code="KNOWLEDGE_BASE_DIR_NOT_FOUND",
            message="知识库目录不存在，无法读取本地知识文档。",
            status_code=500,
        ) from exc
    document_statuses = [
        KnowledgeBaseDocumentStatus(
            source=str(document.metadata["source"]),
            title=str(document.metadata["title"]),
            file_name=str(document.metadata["file_name"]),
            file_extension=str(document.metadata["file_extension"]),
            doc_type=_optional_str(document.metadata.get("doc_type")),
            business_domain=_optional_str(document.metadata.get("business_domain")),
            permission_group=_optional_str(document.metadata.get("permission_group")),
        )
        for document in documents
    ]
    return KnowledgeBaseStatusResponse(
        documents=document_statuses,
        document_count=len(document_statuses),
        collection_name=settings.qdrant_collection_name,
        qdrant_base_url=settings.resolved_qdrant_base_url,
        fake_embedding_dimension=settings.qdrant_vector_size,
        real_embedding_configured=settings.has_embedding_api_key,
        trace_id=get_trace_id(),
    )


@router.post("/ingest", response_model=KnowledgeBaseIngestResponse)
def ingest_knowledge_base(
    request: KnowledgeBaseIngestRequest,
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
    vector_store: VectorStoreUpdater = Depends(get_vector_store),
) -> KnowledgeBaseIngestResponse:
    embedding_model = build_embedding_model(request, settings)

    try:
        if request.refresh:
            result = refresh_directory_in_vector_store(
                directory,
                embedding_model=embedding_model,
                vector_store=vector_store,
                include_readme=request.include_readme,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                wait=request.wait,
            )
        else:
            result = ingest_directory_to_vector_store(
                directory,
                embedding_model=embedding_model,
                vector_store=vector_store,
                include_readme=request.include_readme,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                wait=request.wait,
            )
    except NotADirectoryError as exc:
        raise AppException(
            code="KNOWLEDGE_BASE_DIR_NOT_FOUND",
            message="知识库目录不存在，无法执行入库。",
            status_code=500,
        ) from exc
    except ValueError as exc:
        raise AppException(
            code="KNOWLEDGE_BASE_INGEST_INVALID_CONFIG",
            message=str(exc),
            status_code=400,
        ) from exc

    return KnowledgeBaseIngestResponse(
        embedding_mode=request.embedding_mode,
        document_count=result.document_count,
        chunk_count=result.chunk_count,
        vector_count=result.vector_count,
        vector_dimension=result.vector_dimension,
        collection_name=result.collection_name,
        replaced_source_count=result.replaced_source_count,
        trace_id=get_trace_id(),
    )


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def build_java_document_client(settings: Settings):
    from app.services.java_knowledge_document_client import KnowledgeDocumentClient

    return KnowledgeDocumentClient.from_settings(settings)


def _strip_document_boilerplate(text: str) -> str:
    """剥离 _render_document_markdown 生成的首行标题与元数据行，得到纯正文。"""
    lines = text.splitlines()
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            continue
        if any(
            stripped.startswith(f"{key}:")
            or stripped.startswith(f"> {key}:")
            for key in METADATA_KEY_MAP
        ):
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip()


def _render_document_markdown(
    title: str,
    content: str,
    *,
    business_domain: str,
    permission_group: str,
    doc_type: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"文档类型: {doc_type}",
        f"业务领域: {business_domain}",
        f"权限组: {permission_group}",
        "",
        content.strip(),
        "",
    ]
    return "\n".join(lines)


def _build_single_embedding_model(settings: Settings, mode: str):
    if mode == "fake":
        return DeterministicHashEmbeddingModel(dimension=settings.qdrant_vector_size)
    try:
        return OpenAICompatibleEmbeddingModel.from_settings(settings)
    except ValueError as exc:
        raise AppException(
            code="EMBEDDING_API_KEY_MISSING",
            message="Embedding API key 未配置，无法执行真实 embedding 入库。",
            status_code=500,
        ) from exc


def _validate_document_id(document_id: str) -> None:
    """限制 document_id 为安全字符，防止路径遍历。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", document_id):
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_ID_INVALID",
            message="文档 ID 只能包含字母、数字、下划线、点和连字符。",
            status_code=422,
        )


UPLOAD_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}


def _validate_upload_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_ALLOWED_SUFFIXES:
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_UNSUPPORTED_SUFFIX",
            message=f"不支持的文件类型: {suffix}，仅支持 md/txt/pdf。",
            status_code=400,
        )
    return suffix


def _document_view(
    *,    document_id: str,
    title: str,
    business_domain: str,
    permission_group: str,
    doc_type: str,
    collection_name: str,
    chunk_count: int,
    source_file_name: str,
    exists_local: bool,
    status: str = "enabled",
    updated_at: str | None = None,
) -> KnowledgeBaseDocumentView:
    return KnowledgeBaseDocumentView(
        document_id=document_id,
        title=title,
        business_domain=business_domain,
        permission_group=permission_group,
        doc_type=doc_type,
        collection_name=collection_name,
        chunk_count=chunk_count,
        source_file_name=source_file_name,
        exists_local=exists_local,
        status=status,
        updated_at=updated_at,
    )


@router.get("/documents", response_model=KnowledgeBaseDocumentListView)
def knowledge_base_documents(
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseDocumentListView:
    local_docs: dict[str, dict] = {}
    try:
        documents = load_documents_from_directory(directory)
    except NotADirectoryError:
        documents = []
    for document in documents:
        metadata = document.metadata
        source = str(metadata.get("source") or "")
        local_docs[source] = {
            "source": source,
            "title": str(metadata.get("title") or Path(source).stem),
            "doc_type": _optional_str(metadata.get("doc_type")) or "policy",
            "business_domain": _optional_str(metadata.get("business_domain")) or "general",
            "permission_group": _optional_str(metadata.get("permission_group")) or "public",
        }

    java_client = build_java_document_client(settings)
    java_docs = []
    try:
        java_docs = java_client.list_documents() or []
    except Exception:
        java_docs = []

    views: list[KnowledgeBaseDocumentView] = []
    seen: set[str] = set()
    for java_doc in java_docs:
        document_id = str(java_doc.get("document_id") or "")
        if not document_id:
            continue
        seen.add(document_id)
        source_file_name = str(java_doc.get("source_file_name") or f"{document_id}.md")
        local = local_docs.get(source_file_name) or {}
        views.append(
            _document_view(
                document_id=document_id,
                title=str(java_doc.get("title") or local.get("title") or document_id),
                business_domain=str(java_doc.get("business_domain") or local.get("business_domain") or "general"),
                permission_group=str(java_doc.get("permission_group") or local.get("permission_group") or "public"),
                doc_type=str(java_doc.get("doc_type") or local.get("doc_type") or "policy"),
                collection_name="",
                chunk_count=int(java_doc.get("chunk_count") or 0),
                source_file_name=source_file_name,
                exists_local=source_file_name in local_docs,
                status=str(java_doc.get("status") or "enabled"),
                updated_at=_optional_str(java_doc.get("updated_at")),
            )
        )

    for source, local in local_docs.items():
        if source in seen:
            continue
        views.append(
            _document_view(
                document_id=source,
                title=local["title"],
                business_domain=local["business_domain"],
                permission_group=local["permission_group"],
                doc_type=local["doc_type"],
                collection_name="",
                chunk_count=0,
                source_file_name=source,
                exists_local=True,
            )
        )

    return KnowledgeBaseDocumentListView(
        documents=views,
        document_count=len(views),
        trace_id=get_trace_id(),
    )


@router.post("/documents", response_model=KnowledgeBaseDocumentView)
def create_knowledge_base_document(
    request: KnowledgeBaseDocumentCreateRequest,
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseDocumentView:
    _validate_document_id(request.document_id)
    file_path = directory / f"{request.document_id}.md"
    markdown = _render_document_markdown(
        request.title,
        request.content,
        business_domain=request.business_domain,
        permission_group=request.permission_group,
        doc_type=request.doc_type,
    )
    file_path.write_text(markdown, encoding="utf-8")

    embedding_model = _build_single_embedding_model(settings, request.embedding_mode)
    result = _sync_document_to_all_collections(
        file_path,
        embedding_model=embedding_model,
        settings=settings,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    java_client = build_java_document_client(settings)
    java_client.upsert_document(
        {
            "document_id": request.document_id,
            "title": request.title,
            "doc_type": request.doc_type,
            "business_domain": request.business_domain,
            "permission_group": request.permission_group,
            "status": "enabled",
            "source_file_name": file_path.name,
            "chunk_count": result.chunk_count,
            "updated_by": "ai-service",
        }
    )

    return _document_view(
        document_id=request.document_id,
        title=request.title,
        business_domain=request.business_domain,
        permission_group=request.permission_group,
        doc_type=request.doc_type,
        collection_name=result.collection_name,
        chunk_count=result.chunk_count,
        source_file_name=file_path.name,
        exists_local=True,
        status="enabled",
    )




@router.post("/documents/upload", response_model=KnowledgeBaseDocumentView)
def upload_knowledge_base_document(
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    business_domain: str = Form(default="general"),
    permission_group: str = Form(default="public"),
    embedding_mode: str = Form(default="fake"),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=80),
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseDocumentView:
    if not file.filename:
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_FILE_REQUIRED",
            message="缺少上传文件。",
            status_code=400,
        )
    suffix = _validate_upload_suffix(file.filename)
    if embedding_mode not in {"fake", "real"}:
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_INVALID_EMBEDDING_MODE",
            message="embedding_mode 只能是 fake 或 real。",
            status_code=422,
        )
    resolved_id = (document_id or Path(file.filename).stem).strip()
    _validate_document_id(resolved_id)

    existing = list(directory.glob(f"{resolved_id}.*"))
    if existing:
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_CONFLICT",
            message=f"文档 {resolved_id} 已存在，请删除后重新上传。",
            status_code=409,
        )

    file_path = directory / f"{resolved_id}{suffix}"
    file_path.write_bytes(file.file.read())

    try:
        document = load_document(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)  # 解析失败则清理落盘文件
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_LOAD_FAILED",
            message=f"文档解析失败: {exc}",
            status_code=422,
        ) from exc

    embedding_model = _build_single_embedding_model(settings, embedding_mode)
    result = _sync_document_to_all_collections(
        file_path,
        embedding_model=embedding_model,
        settings=settings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    java_client = build_java_document_client(settings)
    java_client.upsert_document(
        {
            "document_id": resolved_id,
            "title": document.metadata["title"],
            "doc_type": suffix.lstrip("."),
            "business_domain": business_domain,
            "permission_group": permission_group,
            "status": "enabled",
            "source_file_name": file_path.name,
            "chunk_count": result.chunk_count,
            "updated_by": "ai-service",
        }
    )

    return _document_view(
        document_id=resolved_id,
        title=document.metadata["title"],
        business_domain=business_domain,
        permission_group=permission_group,
        doc_type=suffix.lstrip("."),
        collection_name=result.collection_name,
        chunk_count=result.chunk_count,
        source_file_name=file_path.name,
        exists_local=True,
        status="enabled",
    )


@router.put("/documents/{document_id}", response_model=KnowledgeBaseDocumentView)
def update_knowledge_base_document(
    document_id: str,
    request: KnowledgeBaseDocumentUpdateRequest,
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseDocumentView:
    _validate_document_id(document_id)
    file_path = directory / f"{document_id}.md"
    if not file_path.exists():
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_NOT_FOUND",
            message="知识文档不存在。",
            status_code=404,
        )

    existing = load_document(file_path)
    metadata = existing.metadata
    new_title = request.title or str(metadata.get("title") or document_id)
    new_content = (
        request.content
        if request.content and request.content.strip()
        else _strip_document_boilerplate(existing.content)
    )
    business_domain = request.business_domain or _optional_str(metadata.get("business_domain")) or "general"
    permission_group = request.permission_group or _optional_str(metadata.get("permission_group")) or "public"
    doc_type = request.doc_type or _optional_str(metadata.get("doc_type")) or "policy"

    markdown = _render_document_markdown(
        new_title,
        new_content,
        business_domain=business_domain,
        permission_group=permission_group,
        doc_type=doc_type,
    )
    file_path.write_text(markdown, encoding="utf-8")

    embedding_model = _build_single_embedding_model(settings, request.embedding_mode)
    result = _sync_document_to_all_collections(
        file_path,
        embedding_model=embedding_model,
        settings=settings,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    java_client = build_java_document_client(settings)
    java_client.upsert_document(
        {
            "document_id": document_id,
            "title": new_title,
            "doc_type": doc_type,
            "business_domain": business_domain,
            "permission_group": permission_group,
            "status": "enabled",
            "source_file_name": file_path.name,
            "chunk_count": result.chunk_count,
            "updated_by": "ai-service",
        }
    )

    return _document_view(
        document_id=document_id,
        title=new_title,
        business_domain=business_domain,
        permission_group=permission_group,
        doc_type=doc_type,
        collection_name=result.collection_name,
        chunk_count=result.chunk_count,
        source_file_name=file_path.name,
        exists_local=True,
        status="enabled",
    )


@router.delete("/documents/{document_id}", response_model=dict)
def delete_knowledge_base_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> dict:
    _validate_document_id(document_id)
    file_path = directory / f"{document_id}.md"
    if file_path.exists():
        _delete_source_from_all_collections(file_path.name, settings)
        file_path.unlink()

    java_client = build_java_document_client(settings)
    java_client.delete_document(document_id)

    return {"success": True, "trace_id": get_trace_id()}


def _delete_source_from_all_collections(source: str, settings: Settings) -> None:
    """删除 source 在默认库与全部 kb_* collection 中的 chunk。"""
    collection_names = [settings.qdrant_collection_name]
    collection_names.extend(
        sorted({d.collection_name for d in default_rag_knowledge_bases()})
    )
    for collection_name in collection_names:
        try:
            delete_document_from_vector_store(
                source,
                vector_store=build_collection_vector_store(
                    settings, collection_name=collection_name
                ),
                wait=True,
            )
        except Exception:
            # 集合不存在或删除失败不影响主流程，记录即可
            continue


def _sync_document_to_all_collections(
    file_path: Path,
    *,
    embedding_model,
    settings: Settings,
    chunk_size: int,
    chunk_overlap: int,
):
    """从全部 collection 删旧 chunk，再写入默认库（保证一致性）。"""
    source = file_path.name
    _delete_source_from_all_collections(source, settings)
    return update_single_document(
        file_path,
        embedding_model=embedding_model,
        vector_store=build_collection_vector_store(settings),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        wait=True,
    )


@router.post("/documents/{document_id}/ingest", response_model=KnowledgeBaseDocumentView)
def ingest_knowledge_base_document(
    document_id: str,
    request: KnowledgeBaseDocumentIngestRequest,
    settings: Settings = Depends(get_settings),
    directory: Path = Depends(get_knowledge_base_dir),
) -> KnowledgeBaseDocumentView:
    _validate_document_id(document_id)
    file_path = directory / f"{document_id}.md"
    if not file_path.exists():
        raise AppException(
            code="KNOWLEDGE_DOCUMENT_NOT_FOUND",
            message="知识文档不存在。",
            status_code=404,
        )

    existing = load_document(file_path)
    metadata = existing.metadata
    embedding_model = _build_single_embedding_model(settings, request.embedding_mode)
    result = _sync_document_to_all_collections(
        file_path,
        embedding_model=embedding_model,
        settings=settings,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    java_client = build_java_document_client(settings)
    java_client.upsert_document(
        {
            "document_id": document_id,
            "title": str(metadata.get("title") or document_id),
            "doc_type": _optional_str(metadata.get("doc_type")) or "policy",
            "business_domain": _optional_str(metadata.get("business_domain")) or "general",
            "permission_group": _optional_str(metadata.get("permission_group")) or "public",
            "status": "enabled",
            "source_file_name": file_path.name,
            "chunk_count": result.chunk_count,
            "updated_by": "ai-service",
        }
    )

    return _document_view(
        document_id=document_id,
        title=str(metadata.get("title") or document_id),
        business_domain=_optional_str(metadata.get("business_domain")) or "general",
        permission_group=_optional_str(metadata.get("permission_group")) or "public",
        doc_type=_optional_str(metadata.get("doc_type")) or "policy",
        collection_name=result.collection_name,
        chunk_count=result.chunk_count,
        source_file_name=file_path.name,
        exists_local=True,
        status="enabled",
    )
