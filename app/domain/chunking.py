from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping, word-bounded chunks.

    Chunks are measured in characters. Splitting happens on whitespace so
    words are never cut in half. Consecutive chunks share `overlap`
    characters to preserve context across boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for the space that will join words
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Start next chunk with a tail of the previous one (overlap)
            current, current_len = _overlap_tail(current, overlap)
        current.append(word)
        current_len += len(word) + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


def _overlap_tail(words: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing words of `words` fitting within `overlap` chars."""
    tail: list[str] = []
    length = 0
    for word in reversed(words):
        added = len(word) + (1 if tail else 0)
        if length + added > overlap:
            break
        tail.insert(0, word)
        length += added
    return tail, length