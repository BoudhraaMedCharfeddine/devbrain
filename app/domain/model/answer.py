from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Source:
    """A passage used to generate the answer — RAG traceability."""

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: list[Source]