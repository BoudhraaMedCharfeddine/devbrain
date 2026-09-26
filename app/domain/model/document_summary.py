from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """A document enriched with derived stats — for CRUD listing views.

    Kept separate from `Document`: the base entity holds only what the user
    provided at ingestion; `DocumentSummary` adds computed fields (chunk
    count, preview) that belong to the read model, not the write model.
    """

    id: UUID
    title: str
    source: str
    created_at: datetime
    chunk_count: int
    preview: str
