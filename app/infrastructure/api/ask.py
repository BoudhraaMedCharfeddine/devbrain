from __future__ import annotations

from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, status
from google.genai import errors as genai_errors
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
        # Invalid input reached the domain (e.g. empty text after chunking)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except genai_errors.ClientError as exc:
        # 429 rate-limit / quota exhausted / auth error from Gemini
        if exc.code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM rate limit reached, retry later",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM client error: {exc.message}",
        ) from exc
    except genai_errors.ServerError as exc:
        # 5xx from Gemini's side — their upstream failed, not ours
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM upstream error: {exc.message}",
        ) from exc
    except psycopg.OperationalError as exc:
        # DB unreachable (connection refused, timeout, secret missing)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable",
        ) from exc
    except Exception as exc:
        # Last-resort catch: preserves observability, avoids leaking internals
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected error: {type(exc).__name__}",
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
