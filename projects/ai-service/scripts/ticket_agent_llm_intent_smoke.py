import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.ticket_agent import (
    build_ticket_agent_graph,
    build_ticket_agent_input,
    create_llm_ticket_intent_classifier,
)  # noqa: E402
from app.core.exceptions import AppException  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    message = " ".join(args).strip() or "我的订单 1001 到哪了？"
    graph = build_ticket_agent_graph(
        intent_classifier=create_llm_ticket_intent_classifier(),
    )

    try:
        result = graph.invoke(build_ticket_agent_input(message))
    except AppException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": exc.code,
                    "message": exc.message,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "message": message,
                "intent": result.get("intent"),
                "intent_reason": result.get("intent_reason"),
                "node_history": result.get("node_history", []),
                "final_answer": result.get("final_answer"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
