from __future__ import annotations

import httpx


class OllamaLlm:
    """LLM adapter backed by a self-hosted Ollama server (local demos).

    Implements LlmPort. Talks to Ollama's HTTP API at /api/generate.
    Selected via settings.llm_provider = "ollama".
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # Local CPU inference is slow; a generous timeout avoids false failures.
        self._timeout_s = timeout_s

    def generate(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "system": system,
                "prompt": user,
                "stream": False,
                "options": {
                    # Low temperature = grounded answers, same rationale as Gemini
                    "temperature": 0.2,
                },
            },
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
