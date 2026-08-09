# Agent Bad Case Analysis Sample

## Overall

This sample uses synthetic bad cases for Stage 6 lesson 10. It is not a real failure report from the current Agent.

| Item | Value |
| --- | --- |
| Source cases path | synthetic_bad_cases_for_lesson_10 |
| Case filter | synthetic |
| Selected cases | 3 |
| Failed suites | 3 |
| Bad cases | 3 |

## Category Summary

| Category | Count |
| --- | --- |
| agent_routing | 1 |
| rag_retrieval_or_citation | 1 |
| ticket_field_extraction | 1 |

## Analysis Items

### 1. rag / agent_policy_refund_arrival_001

| Item | Value |
| --- | --- |
| Suite | rag |
| Title | RAG + Agent evaluation |
| Priority | p0 |
| Category | rag_retrieval_or_citation |
| Likely layer | RAG retrieval or citation |

#### Evidence

```text
- agent_policy_refund_arrival_001: expected_status=answered actual_status=answered priority=p0
  expected_sources: ['refund-return-policy.md']
  actual_sources: ['account-security-faq.md']
  - missing_sources=['refund-return-policy.md']
```

#### Diagnosis

The Agent did not cite the expected source, cited the wrong source, or failed a RAG source expectation.

#### Review Questions

- Does the knowledge base actually contain the expected source?
- Did retrieval return the right chunk but citation mapping lose the source?
- Is this a retrieval problem or an expected_sources problem?

#### Recommended Action

Check query rewriting, document chunks, source metadata, retrieval threshold, and citation mapping before changing Agent routing.

#### Regression Action

Keep this case in the RAG + Agent eval suite and consider adding a retrieval-only eval case for the same question.

### 2. route / agent_policy_refund_arrival_001

| Item | Value |
| --- | --- |
| Suite | route |
| Title | Agent route evaluation |
| Priority | p0 |
| Category | agent_routing |
| Likely layer | Agent route graph |

#### Evidence

```text
- agent_policy_refund_arrival_001: priority=p0 task_type=policy_question terminal=extract_ticket_fields
  expected_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need
  actual_path: normalize_user_input -> classify_intent -> retrieve_policy -> decide_ticket_need -> extract_ticket_fields
  - visited_forbidden_nodes=['extract_ticket_fields']
```

#### Diagnosis

The Agent visited a wrong node, skipped a required node, or ended at an unexpected terminal node.

#### Review Questions

- What is the first node where actual_path diverges from expected_path?
- Was a forbidden node visited because an earlier state flag was wrong?
- Should the expected route change, or is the graph decision wrong?

#### Recommended Action

Compare expected_path and actual_path, then inspect the node decision that first diverged.

#### Regression Action

Keep this case in the route eval suite and rerun route plus full suite after the graph change.

### 3. field / agent_ticket_logistics_full_001

| Item | Value |
| --- | --- |
| Suite | field |
| Title | Ticket field evaluation |
| Priority | p0 |
| Category | ticket_field_extraction |
| Likely layer | ticket field extraction |

#### Evidence

```text
- agent_ticket_logistics_full_001: priority=p0 task_type=ticket_request field_accuracy=0.5000
  - field order_id expected='ORDER1001' actual=None
```

#### Diagnosis

The Agent did not extract one or more expected ticket fields correctly.

#### Review Questions

- Is the missing field explicitly present or only implied?
- Should the Agent ask a follow-up instead of guessing?
- Would changing extraction logic affect other ticket domains?

#### Recommended Action

Check whether the user message contains the field, whether the expected field is fair, and whether extraction rules or model output schema need adjustment.

#### Regression Action

Keep this case in the field eval suite and add nearby missing-field cases if the boundary is ambiguous.
