# RAG Evaluation Data

This directory stores small, versioned evaluation datasets for local RAG
experiments.

Current files:

```text
retrieval_cases.json  Minimal retrieval metric cases for checking whether retrieval found expected chunks.
rag_cases.json        Broader RAG evaluation design cases with expected behavior, evidence, access, and refusal scenarios.
```

Each retrieval case describes:

- the user `query`;
- the expected source document;
- the expected section or chunk when stable enough;
- optional metadata filters such as `permission_group` and `business_domain`;
- whether the query is expected to return no results.

Stage 4 lesson 38 uses this dataset with `scripts/rag_retrieval_eval.py` to
calculate Hit Rate@K, Recall@K, Precision@K, MRR@K, and bad cases.

Stage 9 lesson 13 adds `rag_cases.json` for designing end-to-end RAG evaluation
coverage before writing answer-quality metrics. These cases describe whether the
system should answer, cite evidence, return no-context, deny access, or block a
security-risk input.

Stage 9 lesson 14 reuses answer and no-context cases from `rag_cases.json` as
retrieval metric cases, so retrieval quality can be measured against the same
expected evidence designed in the broader RAG evaluation set.

Stage 9 lesson 15 reuses the same `rag_cases.json` expectations for deterministic
answer-quality checks: answer points, cited sources, forbidden sources, and
refusal reason codes become separate evaluation dimensions.

Stage 9 lesson 16 combines retrieval metric failures and answer-quality findings
into bad-case root-cause layers, so a failed case can be routed to retrieval,
ranking, generation, citation, refusal, access-control, or security follow-up.

Stage 9 lesson 17 turns those metrics and bad-case layers into parameter tuning
recommendations, making it clearer when to adjust chunk_size, chunk_overlap,
top_k, score_threshold, rerank, prompt, filters, or safety gates.
