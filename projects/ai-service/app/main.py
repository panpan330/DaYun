import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis import Redis

from app.core.config import Settings, get_settings
from app.core.cors import register_cors_middleware
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.telemetry import setup_telemetry, shutdown_telemetry
from app.middleware.rate_limit import register_rate_limit_middleware
from app.middleware.tracing import register_trace_middleware
from app.routers import chat, cost, evaluation, health, knowledge_base, ops_stats, rag, tickets, tools
from app.services.console_agent_service import ConsoleAgentService
from app.services.conversation_sync import PendingConversationSync
from app.services.cost_sync import PendingCostSync
from app.services.java_conversation_client import JavaConversationClient
from app.services.java_cost_client import JavaCostClient


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    service: ConsoleAgentService = app.state.console_agent_service
    sync: PendingConversationSync | None = None
    sync_thread: threading.Thread | None = None
    stop_event = threading.Event()
    cost_sync: PendingCostSync | None = None
    cost_sync_thread: threading.Thread | None = None
    if service.settings.conversation_sync_enabled:
        sync = PendingConversationSync(
            store=service.conversation_store,
            client=JavaConversationClient.from_settings(service.settings),
            redis=Redis.from_url(
                service.settings.resolved_agent_redis_url,
                decode_responses=True,
            ),
            interval_seconds=service.settings.conversation_sync_interval_seconds,
        )
        service.attach_conversation_sync(sync)
        sync_thread = threading.Thread(target=sync.run_forever, args=(stop_event,), daemon=True)
        sync_thread.start()
    if service.settings.cost_sync_enabled:
        cost_sync = PendingCostSync(
            client=JavaCostClient.from_settings(service.settings),
            interval_seconds=service.settings.cost_sync_interval_seconds,
        )
        cost_sync_thread = threading.Thread(
            target=cost_sync.run_forever,
            args=(stop_event,),
            daemon=True,
        )
        cost_sync_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        if sync_thread is not None:
            sync_thread.join(timeout=2)
        if cost_sync_thread is not None:
            cost_sync_thread.join(timeout=2)
        if sync is not None:
            sync.close()
        shutdown_telemetry()
        service.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    setup_telemetry(settings)
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=app_lifespan,
    )
    app.state.console_agent_service = ConsoleAgentService(settings)
    register_exception_handlers(app)
    register_trace_middleware(app)
    register_rate_limit_middleware(app, settings)
    register_cors_middleware(
        app,
        settings.cors_allowed_origin_list,
        settings.normalized_cors_allowed_origin_regex,
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(knowledge_base.router)
    app.include_router(rag.router)
    app.include_router(evaluation.router)
    app.include_router(cost.router)
    app.include_router(tools.router)
    app.include_router(tickets.router)
    app.include_router(ops_stats.router)
    return app


app = create_app()
