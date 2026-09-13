from __future__ import annotations

from functools import lru_cache

from app.application.answer_question import AnswerQuestion
from app.application.ingest_document import IngestDocument
from app.config import Settings, get_settings
from app.domain.port.llm_port import LlmPort
from app.infrastructure.embedding.e5_embedder import E5Embedder
from app.infrastructure.llm.gemini_llm import GeminiLlm
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore


@lru_cache
def get_embedder() -> E5Embedder:
    """Singleton: the e5 model weights (~1.1 GB) must be loaded only once."""
    return E5Embedder()


@lru_cache
def get_vector_store() -> PgVectorStore:
    return PgVectorStore(get_settings().database_url)


@lru_cache
def get_llm() -> LlmPort:
    """Pick the LLM adapter based on settings.llm_provider.

    "gemini" (default, prod) hits the Google API.
    "ollama" (local demos) hits a self-hosted server.
    """
    settings: Settings = get_settings()
    if settings.llm_provider == "gemini":
        return GeminiLlm(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.llm_provider == "ollama":
        # Adapter added at file 5. Fail loudly for now if selected too early.
        from app.infrastructure.llm.ollama_llm import OllamaLlm
        return OllamaLlm(base_url=settings.ollama_base_url, model=settings.ollama_model)
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")


def get_ingest_use_case() -> IngestDocument:
    """Wires the ingestion use case with its adapters."""
    return IngestDocument(embedder=get_embedder(), store=get_vector_store())


def get_answer_use_case() -> AnswerQuestion:
    """Wires the query use case with its adapters."""
    return AnswerQuestion(
        embedder=get_embedder(), store=get_vector_store(), llm=get_llm()
    )