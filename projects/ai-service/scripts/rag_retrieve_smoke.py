from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.rag.embeddings import DeterministicHashEmbeddingModel
from app.rag.retriever import format_retrieved_chunks_for_debug, retrieve_top_k
from app.rag.vector_store import QdrantVectorStore


def main() -> None:
    query = "订单超过 72 小时没有发货怎么办？"
    settings = get_settings()
    embedding_model = DeterministicHashEmbeddingModel(
        dimension=settings.qdrant_vector_size
    )
    vector_store = QdrantVectorStore.from_settings(settings)

    chunks = retrieve_top_k(
        query,
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
        permission_group="customer_service",
        business_domain="order",
        score_threshold=0.2,
    )

    print(f"query: {query}")
    print("retrieved chunks:")
    for line in format_retrieved_chunks_for_debug(chunks):
        print(line)


if __name__ == "__main__":
    main()
