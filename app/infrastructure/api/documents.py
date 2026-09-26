from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from app.infrastructure.api.dependencies import get_vector_store
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentSummaryOut(BaseModel):
    id: UUID
    title: str
    source: str
    created_at: datetime
    chunk_count: int
    preview: str


@router.get("", response_model=list[DocumentSummaryOut])
def list_documents(
    store: PgVectorStore = Depends(get_vector_store),
) -> list[DocumentSummaryOut]:
    summaries = store.list_documents()
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
    summary = store.get_document_with_stats(document_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
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
    deleted = store.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    # 204 No Content: successful deletion, empty body
    return Response(status_code=status.HTTP_204_NO_CONTENT)
