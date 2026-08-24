from fastapi import FastAPI

from app.config import get_settings
from app.infrastructure.api.health import router as health_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Personal RAG assistant — AI Engineering demonstrator",
    )
    app.include_router(health_router)
    return app


app = create_app()