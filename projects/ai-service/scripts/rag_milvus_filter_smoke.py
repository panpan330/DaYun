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

    print("Milvus filter/index smoke test finished")
    print(f"collection: {ingestion_result.collection_name}")
    print(f"chunks: {ingestion_result.chunk_count}")
    print("indexes:")
    for index_name in vector_store.list_indexes():
        print(f"- {index_name}")

    print()
    print("exact metadata filter: permission_group=customer_service, business_domain=refund")
    exact_chunks = retrieve_top_k(
        "退货运费谁承担？",
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=3,
        permission_group="customer_service",
        business_domain="refund",
    )
    for line in format_retrieved_chunks_for_debug(exact_chunks):
        print(line)

    print()
    print("advanced metadata filter: source in [...], chunk_index between 2 and 5")
    query_vector = embedding_model.embed_texts(["退货和物流规则"])[0]
    advanced_chunks = vector_store.query_similar(
        query_vector,
        top_k=5,
        payload_filter={
            "must": [
                {
                    "key": "source",
                    "match": {
                        "any": [
                            "refund-return-policy.md",
                            "order-shipping-policy.md",
                        ]
                    },
                },
                {"key": "chunk_index", "range": {"gte": 2, "lte": 5}},
            ],
            "must_not": [
                {"key": "permission_group", "match": {"value": "internal_only"}},
            ],
        },
    )
    for line in format_retrieved_chunks_for_debug(advanced_chunks):
        print(line)


if __name__ == "__main__":
    main()
