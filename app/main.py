from fastapi import FastAPI

from app.config import get_settings
from app.infrastructure.api.ask import router as ask_router
from app.infrastructure.api.documents import router as documents_router
from app.infrastructure.api.health import router as health_router
from app.infrastructure.api.ingest import router as ingest_router
from app.infrastructure.logging.setup import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level)

    logger = get_logger(__name__)
    logger.info(
        "app_startup",
        app_name=settings.app_name,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Personal RAG assistant — AI Engineering demonstrator",
    )
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(ask_router)
    app.include_router(documents_router)
    return app


app = create_app()
