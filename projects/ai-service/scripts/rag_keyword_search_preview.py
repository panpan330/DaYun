from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.hybrid import SimpleKeywordRetriever
from app.rag.loaders import load_documents_from_directory
from app.rag.splitters import split_documents_into_chunks


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def main() -> None:
    query = "退款多久到账？"
    documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)
    chunks = split_documents_into_chunks(documents, chunk_size=260, chunk_overlap=40)
    retriever = SimpleKeywordRetriever(chunks)
    results = retriever.search(query, top_k=5, permission_group="customer_service")

    print("RAG keyword search preview")
    print(f"query: {query}")
    print(f"chunks: {len(chunks)}")
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown-source")
        section = result.metadata.get("section", "unknown-section")
        matched = ", ".join(result.matched_terms[:8])
        print(
            f"{index}. score={result.score:.4f} source={source} "
            f"section={section} chunk_id={result.chunk_id} matched={matched}"
        )


if __name__ == "__main__":
    main()
