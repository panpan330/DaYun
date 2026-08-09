"""E2E 框架：--run-e2e 开关、三服务自动起停、API 客户端。"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

logger = logging.getLogger(__name__)

# projects/ai-service/tests/e2e/conftest.py
E2E_ROOT = Path(__file__).parent
AI_SERVICE_ROOT = E2E_ROOT.parent.parent  # projects/ai-service
REPO_ROOT = E2E_ROOT.parents[2]  # projects
LOG_DIR = AI_SERVICE_ROOT / ".e2e_logs"

SERVICES = [
    {
        "name": "java",
        "port": 18004,
        "health": "http://127.0.0.1:18004/health",
        "cmd": ["cmd", "/c", "mvn", "-q", "spring-boot:run"],
        "cwd": REPO_ROOT / "java-business-service",
        "timeout": 90,
    },
    {
        "name": "python",
        "port": 8000,
        "health": "http://127.0.0.1:8000/health",
        "cmd": [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "warning",
        ],
        "cwd": AI_SERVICE_ROOT,
        "timeout": 30,
    },
    {
        "name": "mcp",
        "port": 9100,
        # MCP 为 streamable HTTP，无 /health；用端口探测就绪
        "health": None,
        "cmd": [sys.executable, "-m", "app.mcp_servers.product_server"],
        "cwd": AI_SERVICE_ROOT,
        "timeout": 30,
    },
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-e2e", action="store_true", help="Run end-to-end tests against real services")
    parser.addoption("--keep-services", action="store_true", help="Keep services running after tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: requires --run-e2e and real infrastructure")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-e2e"):
        return
    skip = pytest.mark.skip(reason="e2e tests need --run-e2e and real infrastructure")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_ready(spec: dict) -> bool:
    deadline = time.monotonic() + spec["timeout"]
    while time.monotonic() < deadline:
        if spec["health"] is None:
            if _port_open(spec["port"]):
                return True
        else:
            try:
                response = httpx.get(spec["health"], timeout=2)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
        time.sleep(2)
    return False


@pytest.fixture(scope="session", autouse=True)
def services(request: pytest.FixtureRequest) -> None:
    keep = request.config.getoption("--keep-services")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started: list[tuple[subprocess.Popen, Path]] = []
    try:
        for spec in SERVICES:
            if _port_open(spec["port"]):
                logger.info("e2e service %s already running on port %s", spec["name"], spec["port"])
                continue
            log_path = LOG_DIR / f"{spec['name']}.log"
            process = subprocess.Popen(
                spec["cmd"],
                cwd=str(spec["cwd"]),
                stdout=open(log_path, "wb"),
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            started.append((process, log_path))
            if not _wait_ready(spec):
                raise RuntimeError(
                    f"e2e service {spec['name']} failed to become healthy; log: {log_path}"
                )
            logger.info("e2e service %s healthy", spec["name"])
        yield
    finally:
        if keep:
            logger.info("e2e --keep-services: leaving services running")
            return
        for process, _log in started:
            process.terminate()
        for process, _log in started:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
