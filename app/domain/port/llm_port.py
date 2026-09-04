from __future__ import annotations

from typing import Protocol


class LlmPort(Protocol):
    """Generates an answer from a prompt. Implemented at step 4."""

    def generate(self, system: str, user: str) -> str: ...
