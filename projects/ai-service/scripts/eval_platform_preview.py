from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.eval_suite import run_agent_eval_suites  # noqa: E402
from app.evaluation.eval_platform import (  # noqa: E402
    EvalRunContext,
    build_agent_eval_run_snapshot,
    find_eval_dataset_manifest,
    format_eval_dataset_registry,
    load_eval_dataset_registry,
)


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "evaluation" / "datasets.json"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Preview local evaluation dataset registry and optional Agent eval snapshot."
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to the evaluation dataset registry JSON file.",
    )
    parser.add_argument(
        "--agent-snapshot",
        action="store_true",
        help="Run local Agent eval and print a production-style run snapshot.",
    )
    parser.add_argument(
        "--run-id",
        default="local-agent-eval-preview",
        help="Run id used when --agent-snapshot is enabled.",
    )
    parser.add_argument(
        "--candidate-version",
        default="local-working-tree",
        help="Candidate version used when --agent-snapshot is enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    registry = load_eval_dataset_registry(args.registry_path)
    for line in format_eval_dataset_registry(registry):
        print(line)

    if not args.agent_snapshot:
        return 0

    agent_dataset = find_eval_dataset_manifest(
        registry,
        name="agent_eval",
        version="stage6-v1",
    )
    report = run_agent_eval_suites(PROJECT_ROOT / agent_dataset.cases_path)
    snapshot = build_agent_eval_run_snapshot(
        report,
        context=EvalRunContext(
            run_id=args.run_id,
            dataset_name=agent_dataset.name,
            dataset_version=agent_dataset.version,
            candidate_version=args.candidate_version,
            baseline_run_id=agent_dataset.baseline_run_id,
            model_name="fake_or_rule_based",
            prompt_version="local",
            code_version=args.candidate_version,
        ),
    )

    print("")
    print("Agent eval run snapshot")
    print(f"run_id: {snapshot.context.run_id}")
    print(f"dataset: {snapshot.context.dataset_name}:{snapshot.context.dataset_version}")
    print(f"candidate_version: {snapshot.context.candidate_version}")
    print(f"evaluated_checks: {snapshot.evaluated_check_count}")
    print(f"failed_checks: {snapshot.failed_check_count}")
    print(f"passed: {str(snapshot.passed).lower()}")
    for metric in snapshot.metrics:
        print(f"metric.{metric.name}: {metric.value:.6f} ({metric.direction})")
    return 0 if snapshot.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
