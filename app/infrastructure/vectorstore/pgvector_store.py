from __future__ import annotations

from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector

from app.domain.model.chunk import Chunk
from app.domain.model.document_summary import DocumentSummary


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

    def add_document(self, document_id: UUID, title: str, source: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (id, title, source) VALUES (%s, %s, %s)",
                (document_id, title, source),
            )
            conn.commit()

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

    def search(
        self, embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks closest to `embedding` by cosine distance.

        Uses pgvector's `<=>` operator, which the HNSW index built on
        vector_cosine_ops accelerates. The float returned is a similarity
        score in [0, 1]: 1.0 = identical direction, 0.0 = opposite.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, document_id, content, position,
                       embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()

        return [
            (
                Chunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    position=row[3],
                ),
                # cosine distance in [0, 2] -> similarity in [-1, 1] -> clamped [0, 1]
                max(0.0, 1.0 - float(row[4])),
            )
            for row in rows
        ]

    def list_documents(self) -> list[DocumentSummary]:
        """List all documents with their chunk counts and a content preview."""
        from app.domain.model.document_summary import DocumentSummary

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.id,
                    d.title,
                    d.source,
                    d.created_at,
                    COUNT(c.id) AS chunk_count,
                    (
                        SELECT content
                        FROM chunks
                        WHERE document_id = d.id
                        ORDER BY position
                        LIMIT 1
                    ) AS preview
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            )
            rows = cur.fetchall()

        return [
            DocumentSummary(
                id=row[0],
                title=row[1],
                source=row[2],
                created_at=row[3],
                chunk_count=row[4],
                preview=(row[5] or "")[:200],
            )
            for row in rows
        ]

    def get_document_with_stats(self, document_id: UUID) -> DocumentSummary | None:
        """Return one document with its chunk count and preview, or None."""
        from app.domain.model.document_summary import DocumentSummary

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.id,
                    d.title,
                    d.source,
                    d.created_at,
                    COUNT(c.id) AS chunk_count,
                    (
                        SELECT content
                        FROM chunks
                        WHERE document_id = d.id
                        ORDER BY position
                        LIMIT 1
                    ) AS preview
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = %s
                GROUP BY d.id
                """,
                (document_id,),
            )
            row = cur.fetchone()

        if row is None:
            return None
        return DocumentSummary(
            id=row[0],
            title=row[1],
            source=row[2],
            created_at=row[3],
            chunk_count=row[4],
            preview=(row[5] or "")[:200],
        )

    def delete_document(self, document_id: UUID) -> bool:
        """Delete a document (chunks cascade via FK). Returns True if deleted."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            deleted = cur.rowcount
            conn.commit()
        return deleted > 0