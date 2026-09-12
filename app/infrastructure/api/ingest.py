from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.application.ingest_document import IngestDocument
from app.infrastructure.api.dependencies import get_ingest_use_case

router = APIRouter(prefix="/ingest", tags=["ingest"])


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
    try:
        document = usecase.execute(
            title=payload.title, source=payload.source, text=payload.text
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return IngestResponse(id=document.id, title=document.title, source=document.source)
