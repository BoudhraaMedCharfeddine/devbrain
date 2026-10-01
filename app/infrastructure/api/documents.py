from __future__ import annotations

import time
import uuid
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from app.infrastructure.api.dependencies import get_vector_store
from app.infrastructure.logging.setup import get_logger
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)


class DocumentSummaryOut(BaseModel):
    id: UUID
    title: str
    source: str
    created_at: datetime
    chunk_count: int
    preview: str


def _bind_request(route: str, **extra: object) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=str(uuid.uuid4()), route=route, **extra
    )


@router.get("", response_model=list[DocumentSummaryOut])
def list_documents(
    store: PgVectorStore = Depends(get_vector_store),
) -> list[DocumentSummaryOut]:
    _bind_request("/documents GET")
    started = time.perf_counter()
    summaries = store.list_documents()
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info("documents_listed", count=len(summaries), duration_ms=duration_ms)
    return [
        DocumentSummaryOut(
            id=s.id,
            title=s.title,
            source=s.source,
            created_at=s.created_at,
            chunk_count=s.chunk_count,
            preview=s.preview,
        )
        for s in summaries
    ]


@router.get("/{document_id}", response_model=DocumentSummaryOut)
def get_document(
    document_id: UUID,
    store: PgVectorStore = Depends(get_vector_store),
) -> DocumentSummaryOut:
    _bind_request("/documents/{id} GET", document_id=str(document_id))
    summary = store.get_document_with_stats(document_id)
    if summary is None:
        logger.warning("document_not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    logger.info("document_retrieved", chunk_count=summary.chunk_count)
    return DocumentSummaryOut(
        id=summary.id,
        title=summary.title,
        source=summary.source,
        created_at=summary.created_at,
        chunk_count=summary.chunk_count,
        preview=summary.preview,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    store: PgVectorStore = Depends(get_vector_store),
) -> Response:
    _bind_request("/documents/{id} DELETE", document_id=str(document_id))
    deleted = store.delete_document(document_id)
    if not deleted:
        logger.warning("document_not_found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    logger.info("document_deleted")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
