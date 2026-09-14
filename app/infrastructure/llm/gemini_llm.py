from __future__ import annotations

from google import genai
from google.genai import types


class GeminiLlm:
    """LLM adapter backed by Google's Gemini API.

    Implements LlmPort. The client is stateless; we build one per instance
    and keep the model name pinned via config.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        if not api_key:
            raise ValueError("gemini_api_key is required for GeminiLlm")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                # Low temperature keeps answers grounded in the retrieved
                # context; higher values invent more, which is what we
                # specifically want to avoid in RAG.
                temperature=0.2,
            ),
        )
        # `.text` concatenates all text parts; safe for our single-turn use
        return response.text or ""