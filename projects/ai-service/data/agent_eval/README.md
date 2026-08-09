# Agent Evaluation Data

This directory stores small, curated evaluation datasets for the learning
ticket Agent.

Current files:

```text
agent_cases.json  Stage 6 initial Agent evaluation cases.
```

Each case is split into:

- `inputs`: what the user sends to the Agent;
- `expected`: the expected intent, route, fields, tool use, or safety behavior;
- `metadata`: task type, business domain, case type, difficulty, and tags.

Stage 6 lesson 3 designs this dataset. Later lessons will read these cases to
evaluate intent classification, ticket field extraction, Agent routing, RAG
behavior, tool use, confirmation boundaries, and bad cases.

Stage 6 lesson 4 reads `agent_cases.json` with
`app/agents/intent_evaluation.py` and can be run locally with:

```powershell
uv run python scripts/agent_intent_eval.py
```

Stage 6 lesson 5 reads the same dataset with
`app/agents/field_evaluation.py` and can be run locally with:

```powershell
uv run python scripts/agent_ticket_field_eval.py
```

Stage 6 lesson 6 reads the same dataset with
`app/agents/route_evaluation.py` and can be run locally with:

```powershell
uv run python scripts/agent_route_eval.py
```

Stage 6 lesson 7 reads the same dataset with
`app/agents/rag_agent_evaluation.py` and can be run locally with:

```powershell
uv run python scripts/agent_rag_eval.py
```

Stage 6 lesson 8 adds a unified local Agent evaluation suite runner with
`app/agents/eval_suite.py` and can be run locally with:

```powershell
uv run python scripts/agent_eval.py
uv run python scripts/agent_eval.py --suite rag
uv run python scripts/agent_eval.py --list-suites
```

Stage 6 lesson 9 adds Markdown report generation with
`app/agents/eval_report.py`. Run the unified suite and write a stable learning
report with:

```powershell
uv run python scripts/agent_eval.py --cases-path data/agent_eval/agent_cases.json --report-path data/agent_eval/reports/agent_eval_report.md
```

Stage 6 lesson 10 adds bad case analysis with
`app/agents/bad_case_analysis.py`. Run the unified suite and write the current
real bad case analysis report with:

```powershell
uv run python scripts/agent_eval.py --cases-path data/agent_eval/agent_cases.json --bad-case-analysis-path data/agent_eval/reports/agent_bad_case_analysis.md
```

The current real report has no bad cases. `reports/bad_case_analysis_sample.md`
uses synthetic bad cases for lesson 10 and is not a real failure report from the
current Agent.

Stage 6 lesson 11 marks the first 10 P0 cases with `regression` and
`p0_regression` tags. Run the P0 regression suite and write regression reports
with:

```powershell
uv run python scripts/agent_eval.py --cases-path data/agent_eval/agent_cases.json --regression --priority p0 --report-path data/agent_eval/reports/agent_regression_report.md --bad-case-analysis-path data/agent_eval/reports/agent_regression_bad_case_analysis.md
```

The current P0 regression suite selects 10 cases and passes all four suites:
intent, field, route, and RAG + Agent.
