from __future__ import annotations

from sentence_transformers import SentenceTransformer

# e5 models require task-specific prefixes: passages and queries must be
# embedded with different prefixes, otherwise retrieval quality drops sharply.
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


class E5Embedder:
    """Local embedding adapter backed by an intfloat/multilingual-e5 model.

    Implements EmbeddingPort. The model is loaded once at construction and
    kept in memory for the process lifetime.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_PASSAGE_PREFIX + text for text in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([_QUERY_PREFIX + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True => unit vectors, so cosine similarity
        # matches the vector_cosine_ops index we built in pgvector.
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()