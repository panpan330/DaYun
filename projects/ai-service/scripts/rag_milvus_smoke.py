from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.rag.embeddings import DeterministicHashEmbeddingModel
from app.rag.ingestion import ingest_directory_to_vector_store
from app.rag.milvus_store import MilvusVectorStore
from app.rag.retriever import format_retrieved_chunks_for_debug, retrieve_top_k


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def main() -> None:
    query = "退货运费谁承担？"
    settings = get_settings()
    embedding_model = DeterministicHashEmbeddingModel(
        dimension=settings.milvus_vector_size
    )
    vector_store = MilvusVectorStore.from_settings(settings)

    ingestion_result = ingest_directory_to_vector_store(
        KNOWLEDGE_BASE_DIR,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    chunks = retrieve_top_k(
        query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
        permission_group="customer_service",
        business_domain="refund",
    )

    print("Milvus RAG smoke test finished")
    print(f"documents: {ingestion_result.document_count}")
    print(f"chunks: {ingestion_result.chunk_count}")
    print(f"vectors: {ingestion_result.vector_count}")
    print(f"dimension: {ingestion_result.vector_dimension}")
    print(f"collection: {ingestion_result.collection_name}")
    print(f"query: {query}")
    print("retrieved chunks:")
    for line in format_retrieved_chunks_for_debug(chunks):
        print(line)


if __name__ == "__main__":
    main()
