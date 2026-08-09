# Evaluation Registry

This directory records local evaluation dataset manifests.

The registry does not replace the real case files. It describes which evaluation
datasets exist, which version they belong to, where the case file is stored, and
which baseline run should be used for regression comparison.

Current registry:

```text
datasets.json
bad_cases.json
```

Current datasets:

```text
agent_eval:stage6-v1
rag_retrieval_eval:stage9-v1
rag_answer_eval:stage9-v1
```

The case files still live in their original directories:

```text
data/agent_eval/agent_cases.json
data/rag_eval/retrieval_cases.json
data/rag_eval/rag_cases.json
```

Production-oriented evaluation should keep these ideas separate:

- dataset manifest: what dataset and version this is;
- case file: the actual examples and expected results;
- run snapshot: what candidate version was evaluated;
- baseline: which previous run the candidate is compared against;
- regression report: whether the candidate got worse.

`bad_cases.json` is intentionally empty for now because the current checked-in
local Agent regression reports pass. When a future eval run or production issue
finds a real failure, record a sanitized bad case here instead of storing raw
private user content.
