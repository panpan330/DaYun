import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.mcp_clients.minimal_client import collect_minimal_mcp_debug_snapshot


def main() -> None:
    snapshot = asyncio.run(collect_minimal_mcp_debug_snapshot())
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
