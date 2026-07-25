"""Centralised configuration – loaded from environment / .env, never hardcoded."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int = 0) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(_env(key, str(default)))


@dataclass(frozen=True)
class IngestConfig:
    pdf_dpi: int = field(default_factory=lambda: _env_int("PID_PDF_DPI", 300))
    ocr_lang: str = field(default_factory=lambda: _env("PID_OCR_LANG", "eng"))


@dataclass(frozen=True)
class DeltaConfig:
    fuzzy_threshold: int = field(
        default_factory=lambda: _env_int("PID_FUZZY_THRESHOLD", 70)
    )


@dataclass(frozen=True)
class ChatConfig:
    llm_provider: str = field(default_factory=lambda: _env("PID_LLM_PROVIDER", "openai"))
    llm_model: str = field(
        default_factory=lambda: _env("PID_LLM_MODEL", "gpt-4o-mini")
    )
    llm_api_key: str = field(default_factory=lambda: _env("PID_LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: _env("PID_LLM_BASE_URL", ""))
    embedding_model: str = field(
        default_factory=lambda: _env("PID_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(
        default_factory=lambda: _env_int("PID_EMBEDDING_DIM", 1536)
    )
    top_k: int = field(default_factory=lambda: _env_int("PID_TOP_K", 10))


@dataclass(frozen=True)
class WebConfig:
    host: str = field(default_factory=lambda: _env("PID_WEB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PID_WEB_PORT", 7860))
    share: bool = field(default_factory=lambda: _env("PID_WEB_SHARE", "").lower() in ("1", "true", "yes"))


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(
        default_factory=lambda: Path(_env("PID_DATA_DIR", "data"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(_env("PID_OUTPUT_DIR", "output"))
    )
    log_level: str = field(default_factory=lambda: _env("PID_LOG_LEVEL", "INFO"))
    trace_file: str = field(default_factory=lambda: _env("PID_TRACE_FILE", "traces.jsonl"))
    ingest: IngestConfig = field(default_factory=IngestConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    web: WebConfig = field(default_factory=WebConfig)


def load_config() -> Config:
    """Load config from environment. Call once at startup."""
    load_dotenv()  # loads .env file into os.environ
    return Config()
