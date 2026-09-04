from __future__ import annotations

from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector

from app.domain.model.chunk import Chunk


class PgVectorStore:
    """Vector store adapter backed by PostgreSQL + pgvector.

    Implements VectorStorePort using a plain psycopg 3 connection.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._database_url)
        # Teach psycopg how to adapt Python lists <-> pgvector's vector type
        register_vector(conn)
        return conn

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        rows = [
            (chunk.id, chunk.document_id, chunk.content, chunk.position, embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (id, document_id, content, position, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()

    def add_document(self, document_id: UUID, title: str, source: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, title, source) VALUES (%s, %s, %s)",
                (document_id, title, source),
            )
            conn.commit()