from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.domain.model.chunk import Chunk
from app.domain.model.document_summary import DocumentSummary


class VectorStorePort(Protocol):
    """Stores and searches chunks by similarity. Backed by pgvector."""

    def add_document(self, document_id: UUID, title: str, source: str) -> None: ...

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def search(self, embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def get_document_with_stats(self, document_id: UUID) -> DocumentSummary | None: ...

    def delete_document(self, document_id: UUID) -> bool: ...
