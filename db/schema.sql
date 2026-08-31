-- pgvector extension: vector type + similarity operators
CREATE EXTENSION IF NOT EXISTS vector;

-- Ingested documents
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Document chunks with their embedding (the RAG retrieval unit).
-- Dimension 768 = intfloat/multilingual-e5-base
CREATE TABLE IF NOT EXISTS chunks (
    id           UUID PRIMARY KEY,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      TEXT NOT NULL,
    position     INTEGER NOT NULL,
    embedding    vector(768) NOT NULL
);

-- Approximate nearest-neighbour index for cosine similarity search.
-- Pairs with e5 embeddings, which are meant to be compared by cosine.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Fast listing/deletion of a document's chunks
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);