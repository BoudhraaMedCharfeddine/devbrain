from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Chunk:
    """A document fragment — the retrieval unit of the RAG."""

    document_id: UUID
    content: str
    position: int  # index of the chunk within the document
    id: UUID = field(default_factory=uuid4)