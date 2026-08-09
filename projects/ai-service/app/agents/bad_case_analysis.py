from collections import Counter
from pathlib import Path
import re

from pydantic import BaseModel, Field

from app.agents.eval_suite import AgentEvalRunReport, AgentEvalSuiteReport


class BadCaseAnalysisItem(BaseModel):
    suite_name: str = Field(min_length=1)
    suite_title: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    priority: str | None = None
    category: str = Field(min_length=1)
    likely_layer: str = Field(min_length=1)
    diagnosis: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    regression_action: str = Field(min_length=1)
    review_questions: list[str] = Field(default_factory=list)
    evidence_lines: list[str] = Field(default_factory=list)


class BadCaseAnalysisReport(BaseModel):
    source_cases_path: str = Field(min_length=1)
    case_filter: str = "all"
    selected_case_count: int = Field(default=0, ge=0)
    failed_suite_count: int = Field(ge=0)
    bad_case_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    items: list[BadCaseAnalysisItem] = Field(default_factory=list)


_CASE_HEADER_RE = re.compile(r"^- (?P<case_id>[^:]+):")
_PRIORITY_RE = re.compile(r"\bpriority=(?P<priority>p[0-2])\b")


def analyze_agent_eval_bad_cases(
    run_report: AgentEvalRunReport,
) -> BadCaseAnalysisReport:
    items: list[BadCaseAnalysisItem] = []

    for suite_report in run_report.suite_reports:
        if suite_report.passed or _has_no_bad_cases(suite_report.bad_case_lines):
            continue
        for evidence_lines in _split_bad_case_blocks(suite_report.bad_case_lines):
            items.append(_analyze_bad_case_block(suite_report, evidence_lines))

    category_counts = Counter(item.category for item in items)
    return BadCaseAnalysisReport(
        source_cases_path=run_report.cases_path,
        case_filter=run_report.case_filter,
        selected_case_count=run_report.selected_case_count,
        failed_suite_count=run_report.failed_suite_count,
        bad_case_count=len(items),
        category_counts=dict(category_counts),
        items=items,
    )


def build_bad_case_analysis_markdown_report(
    report: BadCaseAnalysisReport,
    *,
    title: str = "Agent Bad Case Analysis",
    note: str | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
    ]
    if note:
        lines.extend([note, ""])

    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Source cases path", report.source_cases_path],
                ["Case filter", report.case_filter],
                ["Selected cases", str(report.selected_case_count)],
                ["Failed suites", str(report.failed_suite_count)],
                ["Bad cases", str(report.bad_case_count)],
            ],
        )
    )

    if report.bad_case_count == 0:
        lines.extend(["", "## Bad Case Analysis", "", "No bad cases to analyze."])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["", "## Category Summary", ""])
    lines.extend(
        _markdown_table(
            ["Category", "Count"],
            [
                [category, str(count)]
                for category, count in sorted(report.category_counts.items())
            ],
        )
    )

    lines.extend(["", "## Analysis Items"])
    for index, item in enumerate(report.items, start=1):
        lines.extend(["", *_bad_case_item_markdown_lines(index, item)])

    return "\n".join(lines).rstrip() + "\n"


def write_bad_case_analysis_markdown_report(
    report: BadCaseAnalysisReport,
    path: Path | str,
    *,
    title: str = "Agent Bad Case Analysis",
    note: str | None = None,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_bad_case_analysis_markdown_report(report, title=title, note=note),
        encoding="utf-8",
    )
    return report_path


def _split_bad_case_blocks(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line == "Bad cases:":
            continue
        if line.startswith("- "):
            if current:
                blocks.append(current)
            current = [line]
            continue
        if current:
            current.append(line)

    if current:
        blocks.append(current)
    return blocks


def _analyze_bad_case_block(
    suite_report: AgentEvalSuiteReport,
    evidence_lines: list[str],
) -> BadCaseAnalysisItem:
    evidence_text = "\n".join(evidence_lines)
    category = _classify_bad_case_category(suite_report.name, evidence_text)
    guidance = _category_guidance(category)

    return BadCaseAnalysisItem(
        suite_name=suite_report.name,
        suite_title=suite_report.title,
        case_id=_extract_case_id(evidence_lines[0]),
        priority=_extract_priority(evidence_text),
        category=category,
        likely_layer=str(guidance["likely_layer"]),
        diagnosis=str(guidance["diagnosis"]),
        recommended_action=str(guidance["recommended_action"]),
        regression_action=str(guidance["regression_action"]),
        review_questions=list(guidance["review_questions"]),
        evidence_lines=evidence_lines,
    )


def _classify_bad_case_category(suite_name: str, evidence_text: str) -> str:
    normalized = evidence_text.lower()
    if "missing_sources" in normalized or (
        suite_name == "rag" and "expected_sources" in normalized
    ):
        return "rag_retrieval_or_citation"
    if "should_create_ticket" in normalized or "ticket_need_source" in normalized:
        return "agent_decision_after_rag"
    if (
        suite_name == "route"
        or "expected_path" in normalized
        or "actual_path" in normalized
        or "forbidden_nodes" in normalized
        or "missing_required_nodes" in normalized
    ):
        return "agent_routing"
    if suite_name == "field" or "field_accuracy" in normalized or "missing_fields" in normalized:
        return "ticket_field_extraction"
    if suite_name == "intent" or "expected=" in normalized or "actual=" in normalized:
        return "intent_classification"
    return "unknown"


def _category_guidance(category: str) -> dict[str, str | list[str]]:
    guidance: dict[str, dict[str, str | list[str]]] = {
        "intent_classification": {
            "likely_layer": "intent classifier",
            "diagnosis": "The Agent selected an intent or route that differs from the expected behavior.",
            "recommended_action": "First verify the expected intent, then review classifier rules, examples, or prompt wording for this message pattern.",
            "regression_action": "Keep this case in the intent eval suite and rerun the full Agent eval suite after the fix.",
            "review_questions": [
                "Is the expected intent correct for the user message?",
                "Does the classifier have enough signal to distinguish this intent?",
                "Would adding a narrower rule create regressions in nearby intents?",
            ],
        },
        "ticket_field_extraction": {
            "likely_layer": "ticket field extraction",
            "diagnosis": "The Agent did not extract one or more expected ticket fields correctly.",
            "recommended_action": "Check whether the user message contains the field, whether the expected field is fair, and whether extraction rules or model output schema need adjustment.",
            "regression_action": "Keep this case in the field eval suite and add nearby missing-field cases if the boundary is ambiguous.",
            "review_questions": [
                "Is the missing field explicitly present or only implied?",
                "Should the Agent ask a follow-up instead of guessing?",
                "Would changing extraction logic affect other ticket domains?",
            ],
        },
        "agent_routing": {
            "likely_layer": "Agent route graph",
            "diagnosis": "The Agent visited a wrong node, skipped a required node, or ended at an unexpected terminal node.",
            "recommended_action": "Compare expected_path and actual_path, then inspect the node decision that first diverged.",
            "regression_action": "Keep this case in the route eval suite and rerun route plus full suite after the graph change.",
            "review_questions": [
                "What is the first node where actual_path diverges from expected_path?",
                "Was a forbidden node visited because an earlier state flag was wrong?",
                "Should the expected route change, or is the graph decision wrong?",
            ],
        },
        "rag_retrieval_or_citation": {
            "likely_layer": "RAG retrieval or citation",
            "diagnosis": "The Agent did not cite the expected source, cited the wrong source, or failed a RAG source expectation.",
            "recommended_action": "Check query rewriting, document chunks, source metadata, retrieval threshold, and citation mapping before changing Agent routing.",
            "regression_action": "Keep this case in the RAG + Agent eval suite and consider adding a retrieval-only eval case for the same question.",
            "review_questions": [
                "Does the knowledge base actually contain the expected source?",
                "Did retrieval return the right chunk but citation mapping lose the source?",
                "Is this a retrieval problem or an expected_sources problem?",
            ],
        },
        "agent_decision_after_rag": {
            "likely_layer": "Agent decision after RAG",
            "diagnosis": "The RAG status and the Agent's follow-up business decision do not match the expected behavior.",
            "recommended_action": "Check the transition from rag_answer_status to ticket_need_source, needs_ticket, issue_type, and confirmation_required.",
            "regression_action": "Keep this case in the RAG + Agent eval suite and add decision-boundary cases for answered vs no_context.",
            "review_questions": [
                "Should answered RAG suppress ticket creation here?",
                "Should no_context become policy_gap ticket creation?",
                "Did state fields after retrieve_policy carry the right status?",
            ],
        },
        "unknown": {
            "likely_layer": "unknown",
            "diagnosis": "The bad case evidence does not match a known analysis category.",
            "recommended_action": "Read the raw evidence and classify the failure manually before changing code.",
            "regression_action": "After manual classification, improve the analyzer or add a more specific eval category.",
            "review_questions": [
                "What exact expectation failed?",
                "Which layer produced the first wrong state?",
                "Is this a product expectation issue or implementation issue?",
            ],
        },
    }
    return guidance[category]


def _bad_case_item_markdown_lines(
    index: int,
    item: BadCaseAnalysisItem,
) -> list[str]:
    lines = [
        f"### {index}. {item.suite_name} / {item.case_id}",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Suite", item.suite_name],
                ["Title", item.suite_title],
                ["Priority", item.priority or "-"],
                ["Category", item.category],
                ["Likely layer", item.likely_layer],
            ],
        )
    )
    lines.extend(
        [
            "",
            "#### Evidence",
            "",
            "```text",
            *item.evidence_lines,
            "```",
            "",
            "#### Diagnosis",
            "",
            item.diagnosis,
            "",
            "#### Review Questions",
            "",
            *[f"- {question}" for question in item.review_questions],
            "",
            "#### Recommended Action",
            "",
            item.recommended_action,
            "",
            "#### Regression Action",
            "",
            item.regression_action,
        ]
    )
    return lines


def _has_no_bad_cases(lines: list[str]) -> bool:
    return lines == ["No bad cases."]


def _extract_case_id(header_line: str) -> str:
    match = _CASE_HEADER_RE.match(header_line)
    if match:
        return match.group("case_id").strip()
    return "unknown_case"


def _extract_priority(evidence_text: str) -> str | None:
    match = _PRIORITY_RE.search(evidence_text)
    if match:
        return match.group("priority")
    return None


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(_table_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[
            "| " + " | ".join(_table_cell(cell) for cell in row) + " |"
            for row in rows
        ],
    ]


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
