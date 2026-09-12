from __future__ import annotations

from functools import lru_cache

from app.application.ingest_document import IngestDocument
from app.config import get_settings
from app.infrastructure.embedding.e5_embedder import E5Embedder
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore


@lru_cache
def get_embedder() -> E5Embedder:
    """Singleton: the e5 model weights (~1.1 GB) must be loaded only once."""
    return E5Embedder()


@lru_cache
def get_vector_store() -> PgVectorStore:
    return PgVectorStore(get_settings().database_url)


def get_ingest_use_case() -> IngestDocument:
    """Wires the ingestion use case with its adapters."""
    return IngestDocument(embedder=get_embedder(), store=get_vector_store())
