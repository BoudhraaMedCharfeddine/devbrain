"""initial schema

Revision ID: 40fb76634f60
Revises:
Create Date: 2026-08-31

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "40fb76634f60"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pgvector extension: vector type + similarity operators
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Ingested documents
    op.execute(
        """
        CREATE TABLE documents (
            id          UUID PRIMARY KEY,
            title       TEXT NOT NULL,
            source      TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # Document chunks with their embedding (the RAG retrieval unit).
    # Dimension 768 = intfloat/multilingual-e5-base
    op.execute(
        """
        CREATE TABLE chunks (
            id           UUID PRIMARY KEY,
            document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content      TEXT NOT NULL,
            position     INTEGER NOT NULL,
            embedding    vector(768) NOT NULL
        )
        """
    )

    # Approximate nearest-neighbour index for cosine similarity search.
    op.execute(
        """
        CREATE INDEX chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )

    # Fast listing/deletion of a document's chunks
    op.execute("CREATE INDEX chunks_document_id_idx ON chunks (document_id)")


def downgrade() -> None:
    # Reverse order: indexes, then tables, then extension
    op.execute("DROP INDEX IF EXISTS chunks_document_id_idx")
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP EXTENSION IF EXISTS vector")