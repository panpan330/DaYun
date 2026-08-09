from argparse import ArgumentParser
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.documents import RetrievedChunk
from app.rag.evaluation import (
    evaluate_retrieval_results,
    format_retrieval_bad_cases,
    format_retrieval_eval_summary,
    load_retrieval_eval_cases,
)
from app.rag.hybrid import KeywordSearchResult, SimpleKeywordRetriever
from app.rag.loaders import load_documents_from_directory
from app.rag.splitters import split_documents_into_chunks


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "rag_eval" / "retrieval_cases.json"
RAG_RETRIEVAL_HISTORY_PATH = PROJECT_ROOT / "data" / "evaluation" / "rag_retrieval_runs.json"
DEFAULT_EVAL_TOP_K = 3
DEFAULT_KEYWORD_MIN_SCORE = 0.2


def main() -> None:
    parser = ArgumentParser(description="Run a local RAG retrieval evaluation smoke.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_EVAL_TOP_K)
    parser.add_argument(
        "--keyword-min-score",
        type=float,
        default=DEFAULT_KEYWORD_MIN_SCORE,
    )
    parser.add_argument(
        "--retriever",
        choices=["keyword", "vector"],
        default="keyword",
        help="keyword (local in-memory chunks) or vector (real Qdrant + embedding)",
    )
    parser.add_argument(
        "--save-run",
        action="store_true",
        default=False,
        help="persist the run to rag_retrieval_runs history",
    )
    args = parser.parse_args()

    cases = load_retrieval_eval_cases(EVAL_CASES_PATH)

    retrievals_by_case_id: dict[str, list[RetrievedChunk]] = {}
    if args.retriever == "keyword":
        documents = load_documents_from_directory(KNOWLEDGE_BASE_DIR)
        chunks = split_documents_into_chunks(documents)
        keyword_retriever = SimpleKeywordRetriever(chunks)
        for eval_case in cases:
            keyword_results = keyword_retriever.search(
                eval_case.query,
                top_k=args.top_k,
                min_score=args.keyword_min_score,
                permission_group=eval_case.permission_group,
                business_domain=eval_case.business_domain,
                doc_type=eval_case.doc_type,
                source=eval_case.source,
            )
            retrievals_by_case_id[eval_case.id] = [
                _keyword_result_to_retrieved_chunk(result)
                for result in keyword_results
            ]
    else:
        # vector mode: real Qdrant + real embedding (manual smoke; not used in automated tests)
        from app.core.config import get_settings
        from app.rag.embeddings import OpenAICompatibleEmbeddingModel
        from app.rag.retriever import retrieve_top_k
        from app.rag.vector_store import QdrantVectorStore

        settings = get_settings()
        embedding_model = OpenAICompatibleEmbeddingModel.from_settings(settings)
        vector_store = QdrantVectorStore.from_settings(settings)
        for eval_case in cases:
            retrievals_by_case_id[eval_case.id] = retrieve_top_k(
                eval_case.query,
                embedding_model=embedding_model,
                vector_store=vector_store,
                top_k=args.top_k,
                permission_group=eval_case.permission_group,
                business_domain=eval_case.business_domain,
                doc_type=eval_case.doc_type,
                source=eval_case.source,
            )

    summary = evaluate_retrieval_results(
        cases,
        retrievals_by_case_id,
        top_k=args.top_k,
    )

    for line in format_retrieval_eval_summary(summary):
        print(line)
    print()
    for line in format_retrieval_bad_cases(summary):
        print(line)

    if args.save_run:
        _persist_run(args, summary, cases)


def _persist_run(args, summary, cases) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.evaluation.rag_retrieval_history import (
        RagRetrievalRun,
        append_rag_retrieval_run,
    )

    run = RagRetrievalRun(
        run_id=f"rag-retrieval-{uuid4().hex[:12]}",
        retriever_mode=args.retriever,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        top_k=args.top_k,
        hit_rate=summary.hit_rate_at_k,
        recall=summary.recall_at_k,
        precision=summary.precision_at_k,
        mrr=summary.mrr_at_k,
        case_count=len(cases),
        results=[_case_result_to_dict(r) for r in (summary.results or [])],
    )
    append_rag_retrieval_run(RAG_RETRIEVAL_HISTORY_PATH, run)

    try:
        from app.core.config import get_settings
        from app.services.java_rag_eval_client import JavaRagEvalClient, JavaRagEvalClientError

        JavaRagEvalClient.from_settings(get_settings()).save_run(run)
        print(f"synced run to java: {run.run_id}")
    except JavaRagEvalClientError as exc:
        print(f"warning: failed to sync run to java: {exc.code} {exc.message}")


def _case_result_to_dict(result) -> dict:
    return {
        "case_id": getattr(result, "case_id", None),
        "hit": getattr(result, "hit", None),
        "recall": getattr(result, "recall_at_k", None),
        "precision": getattr(result, "precision_at_k", None),
        "rank": getattr(result, "reciprocal_rank", None),
    }


def _keyword_result_to_retrieved_chunk(
    result: KeywordSearchResult,
) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=result.chunk_id,
        chunk_id=result.chunk_id,
        content=result.content,
        metadata=result.metadata,
        score=result.score,
    )


if __name__ == "__main__":
    main()
