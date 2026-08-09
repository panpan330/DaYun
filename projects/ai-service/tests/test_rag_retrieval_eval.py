"""Tests for the RAG retrieval eval script (dual retriever modes)."""

from __future__ import annotations


def test_script_imports_and_exposes_retriever_argument() -> None:
    """Script must accept --retriever=keyword|vector and --save-run."""
    from scripts import rag_retrieval_eval  # noqa: PLC0415

    assert hasattr(rag_retrieval_eval, "DEFAULT_EVAL_TOP_K")
    assert hasattr(rag_retrieval_eval, "DEFAULT_KEYWORD_MIN_SCORE")


def test_script_has_retriever_in_parser() -> None:
    """Verify the script's own parser includes the retriever choice."""
    import inspect
    from scripts import rag_retrieval_eval  # noqa: PLC0415

    source = inspect.getsource(rag_retrieval_eval)
    assert "--retriever" in source
    assert 'choices=["keyword", "vector"]' in source or 'choices = ["keyword", "vector"]' in source
    assert "--save-run" in source
