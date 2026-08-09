import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.minimal_graph import run_minimal_graph, stream_minimal_graph_updates


if __name__ == "__main__":
    user_message = "  你好，LangGraph  "
    result = {
        "invoke_result": run_minimal_graph(user_message),
        "stream_updates": stream_minimal_graph_updates(user_message),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
