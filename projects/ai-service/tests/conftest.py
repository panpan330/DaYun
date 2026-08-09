from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.schemas.tool import ToolAccessLevel, ToolDefinition  # noqa: E402
from app.tools import tool_registry as tool_registry_module  # noqa: E402
from app.tools.tool_confirmation import clear_tool_confirmation_store  # noqa: E402
from app.tools.idempotency import clear_idempotency_store  # noqa: E402


@pytest.fixture
def disabled_sensitive_tool(monkeypatch: pytest.MonkeyPatch) -> str:
    """Registers a temporary disabled sensitive tool in the tool registry.

    The production registry ships with every tool enabled, so tests that
    exercise the "disabled tool is rejected" guard register their own disabled
    definition named ``refund_order_disabled``.
    """

    definition = ToolDefinition(
        name="refund_order_disabled",
        description="temporary disabled sensitive tool for tests",
        access_level=ToolAccessLevel.SENSITIVE,
        requires_confirmation=True,
        enabled=False,
    )
    monkeypatch.setitem(
        tool_registry_module.TOOL_REGISTRY,
        "refund_order_disabled",
        definition,
    )
    return "refund_order_disabled"


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_tool_idempotency_store() -> None:
    clear_idempotency_store()
    yield
    clear_idempotency_store()


@pytest.fixture(autouse=True)
def clear_pending_tool_confirmations() -> None:
    clear_tool_confirmation_store()
    yield
    clear_tool_confirmation_store()


@pytest.fixture
def app() -> FastAPI:
    settings = Settings(_env_file=None)
    test_app = create_app(settings)
    test_app.dependency_overrides[get_settings] = lambda: settings
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)
