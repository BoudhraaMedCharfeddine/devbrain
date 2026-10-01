from __future__ import annotations

import time
import uuid
from uuid import UUID

import psycopg
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from google.genai import errors as genai_errors
from pydantic import BaseModel, Field

from app.application.answer_question import NO_ANSWER, AnswerQuestion
from app.domain.model.query import Query
from app.infrastructure.api.dependencies import get_answer_use_case
from app.infrastructure.logging.setup import get_logger

router = APIRouter(prefix="/ask", tags=["ask"])
logger = get_logger(__name__)


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
    # Bind per-request context: every log in this request carries request_id + route.
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        route="/ask",
        top_k=payload.top_k,
        question_length=len(payload.question),
    )
    logger.info("ask_started")

    started = time.perf_counter()
    try:
        result = usecase.execute(Query(text=payload.question, top_k=payload.top_k))
    except ValueError as exc:
        logger.warning("ask_rejected", reason="invalid_input", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            logger.warning("ask_failed", reason="llm_rate_limit", code=exc.code)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM rate limit reached, retry later",
            ) from exc
        logger.error("ask_failed", reason="llm_client_error", code=exc.code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM client error: {exc.message}",
        ) from exc
    except genai_errors.ServerError as exc:
        logger.error("ask_failed", reason="llm_server_error", code=exc.code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM upstream error: {exc.message}",
        ) from exc
    except psycopg.OperationalError as exc:
        logger.error("ask_failed", reason="vector_store_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("ask_failed", reason="unexpected", exc_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unexpected error: {type(exc).__name__}",
        ) from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    short_circuited = not result.sources
    logger.info(
        "ask_completed",
        duration_ms=duration_ms,
        sources_count=len(result.sources),
        source_scores=[round(s.score, 3) for s in result.sources],
        source_chunk_ids=[str(s.chunk_id) for s in result.sources],
        short_circuited=short_circuited,
        answer_length=len(result.text),
        answer_is_no_answer=result.text == NO_ANSWER,
    )

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
