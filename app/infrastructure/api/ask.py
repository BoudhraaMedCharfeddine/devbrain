from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.application.answer_question import AnswerQuestion
from app.domain.model.query import Query
from app.infrastructure.api.dependencies import get_answer_use_case

router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceOut(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@router.post("", response_model=AskResponse, status_code=status.HTTP_200_OK)
def ask(
    payload: AskRequest,
    usecase: AnswerQuestion = Depends(get_answer_use_case),
) -> AskResponse:
    try:
        result = usecase.execute(Query(text=payload.question, top_k=payload.top_k))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        # LLM/DB errors bubble up as 502 (bad upstream) rather than 500
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM or vector store error: {exc}",
        ) from exc

    return AskResponse(
        answer=result.text,
        sources=[
            SourceOut(
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                content=s.content,
                score=s.score,
            )
            for s in result.sources
        ],
    )
