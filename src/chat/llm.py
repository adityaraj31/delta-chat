"""Provider-agnostic LLM client."""

from __future__ import annotations

from typing import Any

import httpx

from src.config import ChatConfig


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(self, config: ChatConfig | None = None) -> None:
        self._cfg = config or ChatConfig()
        self._base_url = self._cfg.llm_base_url.rstrip("/") if self._cfg.llm_base_url else "https://api.openai.com/v1"
        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._cfg.llm_api_key:
            self._headers["Authorization"] = f"Bearer {self._cfg.llm_api_key}"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message."""
        payload: dict[str, Any] = {
            "model": self._cfg.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        payload: dict[str, Any] = {
            "model": self._cfg.embedding_model,
            "input": texts,
        }
        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
