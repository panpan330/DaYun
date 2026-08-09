import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.ticket_agent import (  # noqa: E402
    build_ticket_agent_graph,
    build_ticket_agent_input,
    create_llm_ticket_field_extractor,
)
from app.core.exceptions import AppException  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    message = " ".join(args).strip() or "商品破损，订单 A2001，帮我投诉处理"
    graph = build_ticket_agent_graph(
        field_extractor=create_llm_ticket_field_extractor(),
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
                "ticket_fields": result.get("ticket_fields"),
                "missing_ticket_fields": result.get("missing_ticket_fields"),
                "ticket_fields_complete": result.get("ticket_fields_complete"),
                "ticket_field_extraction_source": result.get(
                    "ticket_field_extraction_source"
                ),
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
