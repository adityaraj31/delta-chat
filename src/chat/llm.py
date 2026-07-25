"""Provider-agnostic LLM client using langchain framework."""

from __future__ import annotations

import time
import structlog
from src.config import ChatConfig

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    raise ImportError("Langchain framework is required. Please install 'langchain' and 'langchain-openai'.")

log = structlog.get_logger(__name__)

class LLMClient:
    """Thin wrapper around the LangChain ChatOpenAI client, compatible with Langfuse tracing."""

    def __init__(self, config: ChatConfig | None = None) -> None:
        self._cfg = config or ChatConfig()
        self._base_url = self._cfg.llm_base_url.rstrip("/") if self._cfg.llm_base_url else "https://api.openai.com/v1"
        self._chat = ChatOpenAI(
            api_key=self._cfg.llm_api_key or "dummy",
            base_url=self._base_url,
            model=self._cfg.llm_model,
        )
        self._embeddings = OpenAIEmbeddings(
            api_key=self._cfg.llm_api_key or "dummy",
            base_url=self._base_url,
            model=self._cfg.embedding_model,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message."""
        lc_messages = []
        for m in messages:
            if m["role"] == "system": lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user": lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant": lc_messages.append(AIMessage(content=m["content"]))

        start = time.monotonic()
        try:
            # Note: We omit the Langfuse CallbackHandler here because we already 
            # trace the outer 'chat.answer' via the @observe decorator in answer.py, 
            # so we avoid double-tracing the same generation.
            resp = self._chat.invoke(
                lc_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            
            output_text = str(resp.content)
            
            # Extract token usage from the response metadata if present
            usage = resp.response_metadata.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
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
        """Get embeddings for a list of texts. Batches automatically via Langchain."""
        start = time.monotonic()
        try:
            embeddings = self._embeddings.embed_documents(texts)
            elapsed_ms = round((time.monotonic() - start) * 1000, 1)
            
            log.info(
                "llm.embed",
                model=self._cfg.embedding_model,
                elapsed_ms=elapsed_ms,
                batch_size=len(texts),
                total=len(embeddings),
            )
            return embeddings
        except Exception as exc:
            log.exception("llm.embed.error", model=self._cfg.embedding_model, error=str(exc))
            raise

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._embeddings.embed_query(text)
