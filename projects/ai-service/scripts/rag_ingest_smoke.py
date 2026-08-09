from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.rag.embeddings import DeterministicHashEmbeddingModel
from app.rag.ingestion import ingest_directory_to_vector_store
from app.rag.vector_store import QdrantVectorStore

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def main() -> None:
    settings = get_settings()
    embedding_model = DeterministicHashEmbeddingModel(
        dimension=settings.qdrant_vector_size
    )
    vector_store = QdrantVectorStore.from_settings(settings)

    result = ingest_directory_to_vector_store(
        KNOWLEDGE_BASE_DIR,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    print("RAG ingestion smoke test finished")
    print(f"documents: {result.document_count}")
    print(f"chunks: {result.chunk_count}")
    print(f"vectors: {result.vector_count}")
    print(f"dimension: {result.vector_dimension}")
    print(f"collection: {result.collection_name}")


if __name__ == "__main__":
    main()
