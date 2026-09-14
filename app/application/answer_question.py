from __future__ import annotations

from app.domain.model.answer import Answer, Source
from app.domain.model.query import Query
from app.domain.port.embedding_port import EmbeddingPort
from app.domain.port.llm_port import LlmPort
from app.domain.port.vector_store_port import VectorStorePort

_SYSTEM_PROMPT = """You are DevBrain, a technical assistant answering
questions strictly from the SOURCES provided by the user. Rules:

1. If the SOURCES do not contain the answer, reply exactly:
   "Je ne trouve pas la réponse dans les sources fournies."
   Never invent facts, code, or references.
2. When citing, refer to sources by their number in square brackets, e.g. [1], [2].
3. Answer concisely, in the same language as the question.
4. Prefer quoting the sources over paraphrasing when accuracy matters."""


def _build_user_prompt(question: str, sources: list[Source]) -> str:
    """Assemble the retrieved chunks and the question into a single prompt."""
    if not sources:
        return f"SOURCES:\n(none)\n\nQUESTION:\n{question}"

    blocks = [
        f"[{i + 1}] (score={src.score:.3f})\n{src.content}"
        for i, src in enumerate(sources)
    ]
    joined = "\n\n".join(blocks)
    return f"SOURCES:\n{joined}\n\nQUESTION:\n{question}"


class AnswerQuestion:
    """Query use case: question -> retrieval -> generation with citations."""

    def __init__(
        self, embedder: EmbeddingPort, store: VectorStorePort, llm: LlmPort
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm

    def execute(self, query: Query) -> Answer:
        # 1. Embed the question with the `query:` e5 prefix (adapter handles it)
        q_vector = self._embedder.embed_query(query.text)

        # 2. Retrieve the top_k most similar chunks
        hits = self._store.search(q_vector, top_k=query.top_k)
        sources = [
            Source(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score,
            )
            for chunk, score in hits
        ]

        # 3. Ask the LLM to answer grounded in those chunks
        user_prompt = _build_user_prompt(query.text, sources)
        answer_text = self._llm.generate(system=_SYSTEM_PROMPT, user=user_prompt)

        return Answer(text=answer_text, sources=sources)
