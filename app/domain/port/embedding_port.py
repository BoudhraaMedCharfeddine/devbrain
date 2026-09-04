from __future__ import annotations

from typing import Protocol


class EmbeddingPort(Protocol):
    """Turns text into vectors. Implemented by an embedding adapter."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
