"""Provider-agnostic LLM client."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from src.config import ChatConfig

log = structlog.get_logger(__name__)


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
        start = time.monotonic()
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        log.info(
            "llm.chat",
            model=self._cfg.llm_model,
            elapsed_ms=elapsed_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return str(data["choices"][0]["message"]["content"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts. Batches automatically."""
        _BATCH_SIZE = 1000
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            payload: dict[str, Any] = {
                "model": self._cfg.embedding_model,
                "input": batch,
            }
            start = time.monotonic()
            resp = httpx.post(
                f"{self._base_url}/embeddings",
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            if not resp.is_success:
                log.error("llm.embed.error", status=resp.status_code, elapsed_ms=elapsed_ms, batch_size=len(batch))
                raise RuntimeError(
                    f"Embedding API error {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend(item["embedding"] for item in sorted_data)
            log.info(
                "llm.embed",
                model=self._cfg.embedding_model,
                elapsed_ms=elapsed_ms,
                batch_size=len(batch),
                total=len(all_embeddings),
            )
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
