from __future__ import annotations

import time
import uuid
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.application.ingest_document import IngestDocument
from app.infrastructure.api.dependencies import get_ingest_use_case
from app.infrastructure.logging.setup import get_logger

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(__name__)


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    id: UUID
    title: str
    source: str


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def ingest(
    payload: IngestRequest,
    usecase: IngestDocument = Depends(get_ingest_use_case),
) -> IngestResponse:
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        route="/ingest",
        title=payload.title,
        source=payload.source,
        text_length=len(payload.text),
    )
    logger.info("ingest_started")

    started = time.perf_counter()
    try:
        document = usecase.execute(
            title=payload.title, source=payload.source, text=payload.text
        )
    except ValueError as exc:
        logger.warning("ingest_rejected", reason="invalid_input", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("ingest_failed", exc_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected error: {type(exc).__name__}",
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "ingest_completed",
        duration_ms=duration_ms,
        document_id=str(document.id),
    )

    return IngestResponse(id=document.id, title=document.title, source=document.source)
