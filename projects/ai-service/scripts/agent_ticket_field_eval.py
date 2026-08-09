from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.field_evaluation import (  # noqa: E402
    evaluate_ticket_field_cases,
    format_ticket_field_bad_cases,
    format_ticket_field_eval_summary,
)
from app.agents.intent_evaluation import load_agent_eval_cases  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "agent_eval" / "agent_cases.json"


def main() -> int:
    cases = load_agent_eval_cases(DEFAULT_CASES_PATH)
    summary = evaluate_ticket_field_cases(cases)

    for line in format_ticket_field_eval_summary(summary):
        print(line)
    for line in format_ticket_field_bad_cases(summary):
        print(line)

    return 0 if summary.failed_case_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
