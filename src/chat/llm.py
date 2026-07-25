"""Provider-agnostic LLM client."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import httpx
import structlog

from src.config import ChatConfig
from src.observability.tracing import estimate_cost_usd, get_tracer

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
        tracer = get_tracer()
        with tracer.span(
            "llm.chat",
            as_type="generation",
            input={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            model=self._cfg.llm_model,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as generation:
            try:
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
                output_text = str(data["choices"][0]["message"]["content"])
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                cost = estimate_cost_usd(self._cfg.llm_model, prompt_tokens, completion_tokens)

                generation.update(
                    output=output_text,
                    metadata={
                        "elapsed_ms": elapsed_ms,
                        "status_code": resp.status_code,
                    },
                    usage_details={
                        "prompt_tokens": int(prompt_tokens or 0),
                        "completion_tokens": int(completion_tokens or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                    cost_details={"total_cost": cost or 0.0},
                    completion_start_time=datetime.now(timezone.utc),
                )
                log.info(
                    "llm.chat",
                    model=self._cfg.llm_model,
                    elapsed_ms=elapsed_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=usage.get("total_tokens"),
                    estimated_cost_usd=cost,
                )
                return output_text
            except Exception as exc:
                generation.update(
                    metadata={"error": str(exc)},
                    cost_details={"total_cost": 0.0},
                )
                log.exception("llm.chat.error", model=self._cfg.llm_model, error=str(exc))
                raise

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
            tracer = get_tracer()
            with tracer.span(
                "llm.embed",
                as_type="embedding",
                input=batch,
                model=self._cfg.embedding_model,
            ) as embedding_obs:
                try:
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
                    embeddings = [item["embedding"] for item in sorted_data]
                    all_embeddings.extend(embeddings)
                    usage = data.get("usage", {})
                    embedding_obs.update(
                        output={"embeddings": len(embeddings), "batch_size": len(batch)},
                        metadata={
                            "elapsed_ms": elapsed_ms,
                            "status_code": resp.status_code,
                        },
                        usage_details={
                            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                            "total_tokens": int(usage.get("total_tokens") or 0),
                        },
                        cost_details={"total_cost": estimate_cost_usd(self._cfg.embedding_model, usage.get("prompt_tokens"), None) or 0.0},
                    )
                    log.info(
                        "llm.embed",
                        model=self._cfg.embedding_model,
                        elapsed_ms=elapsed_ms,
                        batch_size=len(batch),
                        total=len(all_embeddings),
                    )
                except Exception as exc:
                    embedding_obs.update(metadata={"error": str(exc)})
                    log.exception("llm.embed.error", model=self._cfg.embedding_model, error=str(exc))
                    raise
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
