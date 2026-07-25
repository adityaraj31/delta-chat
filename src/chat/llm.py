"""Provider-agnostic LLM client using openai sdk and langfuse wrapper."""

from __future__ import annotations

import time

import structlog
from langfuse.openai import openai

from src.config import ChatConfig

log = structlog.get_logger(__name__)

class LLMClient:
    """Thin wrapper around the OpenAI client, automatically traced by Langfuse."""

    def __init__(self, config: ChatConfig | None = None) -> None:
        self._cfg = config or ChatConfig()
        self._base_url = self._cfg.llm_base_url.rstrip("/") if self._cfg.llm_base_url else "https://api.openai.com/v1"
        self._client = openai.OpenAI(
            api_key=self._cfg.llm_api_key or "dummy",
            base_url=self._base_url
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message."""
        start = time.monotonic()
        try:
            # pyrefly: ignore [no-matching-overload]
            resp = self._client.chat.completions.create(
                model=self._cfg.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                name="llm.chat"
            )
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            output_text = str(resp.choices[0].message.content)
            
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage is not None else 0
            completion_tokens = usage.completion_tokens if usage is not None else 0
            total_tokens = usage.total_tokens if usage is not None else 0
            
            log.info(
                "llm.chat",
                model=self._cfg.llm_model,
                elapsed_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            return output_text
        except Exception as exc:
            log.exception("llm.chat.error", model=self._cfg.llm_model, error=str(exc))
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts. Batches automatically."""
        _BATCH_SIZE = 1000
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            start = time.monotonic()
            try:
                resp = self._client.embeddings.create(
                    model=self._cfg.embedding_model,
                    input=batch,
                    # pyrefly: ignore [unexpected-keyword]
                    name="llm.embed"
                )
                elapsed_ms = round((time.monotonic() - start) * 1000, 1)
                
                # Sort back into requested order if necessary, OpenAI usually preserves order
                embeddings = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
                all_embeddings.extend(embeddings)
                log.info(
                    "llm.embed",
                    model=self._cfg.embedding_model,
                    elapsed_ms=elapsed_ms,
                    batch_size=len(batch),
                    total=len(all_embeddings),
                )
            except Exception as exc:
                log.exception("llm.embed.error", model=self._cfg.embedding_model, error=str(exc))
                raise
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
