from app.core.exceptions import AppException
from app.rag.vector_store import QdrantCollectionConfigError, QdrantVectorStoreError


def rag_embedding_failed(exc: Exception) -> AppException:
    if isinstance(exc, AppException):
        return exc
    if isinstance(exc, ValueError):
        return rag_embedding_bad_response()
    return AppException(
        code="RAG_EMBEDDING_FAILED",
        message="RAG embedding 生成失败，请稍后重试。",
        status_code=502,
    )


def rag_embedding_bad_response() -> AppException:
    return AppException(
        code="RAG_EMBEDDING_BAD_RESPONSE",
        message="RAG embedding 返回结果结构异常。",
        status_code=502,
    )


def rag_vector_store_failed(exc: Exception) -> AppException:
    if isinstance(exc, AppException):
        return exc
    if isinstance(exc, QdrantCollectionConfigError):
        return AppException(
            code="RAG_VECTOR_STORE_CONFIG_ERROR",
            message="RAG 向量库 collection 配置与当前 embedding 配置不一致。",
            status_code=500,
        )
    if isinstance(exc, QdrantVectorStoreError):
        return AppException(
            code="RAG_VECTOR_STORE_FAILED",
            message="RAG 向量库调用失败，请稍后重试。",
            status_code=502,
        )
    return AppException(
        code="RAG_VECTOR_STORE_FAILED",
        message="RAG 向量库调用失败，请稍后重试。",
        status_code=502,
    )
