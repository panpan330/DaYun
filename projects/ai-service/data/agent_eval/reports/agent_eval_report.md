# Agent Evaluation Report

## Overall

| Item | Value |
| --- | --- |
| Status | PASS |
| Cases path | data\agent_eval\agent_cases.json |
| Case filter | all |
| Selected cases | 12 |
| Suites | intent, field, route, rag |
| Suite count | 4 |
| Passed suites | 4 |
| Failed suites | 0 |

## Suite Summary

| Suite | Title | Cases | Failed cases | Status |
| --- | --- | --- | --- | --- |
| intent | Intent evaluation | 12 | 0 | PASS |
| field | Ticket field evaluation | 4 | 0 | PASS |
| route | Agent route evaluation | 12 | 0 | PASS |
| rag | RAG + Agent evaluation | 3 | 0 | PASS |

## intent: Intent evaluation

### Summary

```text
Agent intent evaluation summary
cases: 12
passed_cases: 12
failed_cases: 0
accuracy: 1.0000
p0_cases: 10
p0_passed_cases: 10
p0_failed_cases: 0
p0_accuracy: 1.0000
```

### Bad Cases

```text
No bad cases.
```

## field: Ticket field evaluation

### Summary

```text
Agent ticket field evaluation summary
cases: 4
passed_cases: 4
failed_cases: 0
case_pass_rate: 1.0000
expected_fields: 16
matched_fields: 16
field_accuracy: 1.0000
p0_cases: 4
p0_passed_cases: 4
p0_failed_cases: 0
p0_case_pass_rate: 1.0000
missing_field_cases: 1
missing_field_passed_cases: 1
```

### Bad Cases

```text
No bad cases.
```

## route: Agent route evaluation

### Summary

```text
Agent route evaluation summary
cases: 12
passed_cases: 12
failed_cases: 0
route_pass_rate: 1.0000
exact_match_count: 12
exact_match_rate: 1.0000
required_nodes_passed_count: 12
forbidden_nodes_passed_count: 12
terminal_node_passed_count: 12
p0_cases: 10
p0_passed_cases: 10
p0_failed_cases: 0
p0_route_pass_rate: 1.0000
```

### Bad Cases

```text
No bad cases.
```

## rag: RAG + Agent evaluation

### Summary

```text
RAG + Agent evaluation summary
cases: 3
passed_cases: 3
failed_cases: 0
case_pass_rate: 1.0000
answered_cases: 2
answered_passed_cases: 2
no_context_cases: 1
no_context_passed_cases: 1
expected_sources: 2
matched_sources: 2
source_recall: 1.0000
citation_passed_count: 3
ticket_decision_passed_count: 3
p0_cases: 3
p0_passed_cases: 3
p0_failed_cases: 0
p0_case_pass_rate: 1.0000
```

### Bad Cases

```text
No bad cases.
```
