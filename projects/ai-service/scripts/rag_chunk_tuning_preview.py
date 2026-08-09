from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.loaders import load_documents_from_directory
from app.rag.tuning import ChunkTuningCase, compare_chunk_tuning_cases


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def main() -> None:
    documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)
    reports = compare_chunk_tuning_cases(
        documents,
        [
            ChunkTuningCase(chunk_size=180, chunk_overlap=20),
            ChunkTuningCase(chunk_size=260, chunk_overlap=40),
            ChunkTuningCase(chunk_size=420, chunk_overlap=80),
        ],
    )

    print("RAG chunk tuning preview")
    print(f"documents: {len(documents)}")
    for report in reports:
        print(
            "chunk_size={chunk_size} overlap={overlap} chunks={chunks} "
            "min={min_chars} max={max_chars} avg={avg_chars}".format(
                chunk_size=report.chunk_size,
                overlap=report.chunk_overlap,
                chunks=report.chunk_count,
                min_chars=report.min_chunk_chars,
                max_chars=report.max_chunk_chars,
                avg_chars=report.average_chunk_chars,
            )
        )


if __name__ == "__main__":
    main()
