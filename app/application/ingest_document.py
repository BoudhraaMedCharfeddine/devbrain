from __future__ import annotations

from app.domain.chunking import chunk_text
from app.domain.model.chunk import Chunk
from app.domain.model.document import Document
from app.domain.port.embedding_port import EmbeddingPort
from app.domain.port.vector_store_port import VectorStorePort


class IngestDocument:
    """Ingestion use case: document -> chunks -> embeddings -> storage."""

    def __init__(self, embedder: EmbeddingPort, store: VectorStorePort) -> None:
        self._embedder = embedder
        self._store = store

    def execute(self, title: str, source: str, text: str) -> Document:
        document = Document(title=title, source=source)

        pieces = chunk_text(text)
        if not pieces:
            raise ValueError("document text produced no chunks")

        chunks = [
            Chunk(document_id=document.id, content=piece, position=i)
            for i, piece in enumerate(pieces)
        ]
        embeddings = self._embedder.embed_documents([c.content for c in chunks])

        # Parent row first (chunks reference it via FK), then the chunks
        self._store.add_document(document.id, document.title, document.source)
        self._store.add_chunks(chunks, embeddings)

        return document