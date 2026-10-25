"""Settings read from the environment (and .env if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class ModelSpec:
    """One model Tempo can route to."""

    name: str  # Tempo's name for it, used in plans and receipts
    provider: str  # "ollama" | "gemini" | "fake"
    model_id: str  # provider's model string
    location: str  # "local" | "cloud"
    tier: int  # 0 = smallest; escalation goes up the tiers
    cutoff: date  # knowledge cutoff, read by the freshness signal
    price_in: float = 0.0  # USD per 1M input tokens (list price)
    price_out: float = 0.0  # USD per 1M output tokens (list price)


@dataclass(frozen=True)
class Settings:
    gemini_keys: list[str] = field(default_factory=list)
    gemini_light: str = "gemini-2.5-flash-lite"
    gemini_large: str = "gemini-2.5-flash"
    openrouter_key: str = ""  # set = cloud tiers go through OpenRouter instead of Gemini direct
    openrouter_light: str = "google/gemini-2.5-flash-lite"
    openrouter_large: str = "google/gemini-2.5-flash"
    openrouter_embed: str = "google/gemini-embedding-001"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embed_provider: str = "gemini"  # gemini | openrouter | ollama | hash
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    tavily_key: str = ""
    database_url: str = ""
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    confidence_threshold: float = 0.6
    retrieval_threshold: float = 0.55

    @property
    def cloud_provider(self) -> str:
        return "openrouter" if self.openrouter_key else "gemini"

    @property
    def models(self) -> list[ModelSpec]:
        return [
            ModelSpec(
                name="local",
                provider="ollama",
                model_id=self.ollama_model,
                location="local",
                tier=0,
                cutoff=date(2023, 12, 1),
            ),
            ModelSpec(
                name="light",
                provider=self.cloud_provider,
                model_id=self.openrouter_light if self.openrouter_key else self.gemini_light,
                location="cloud",
                tier=1,
                cutoff=date(2025, 1, 1),
                price_in=_float("PRICE_LIGHT_IN", 0.10),
                price_out=_float("PRICE_LIGHT_OUT", 0.40),
            ),
            ModelSpec(
                name="large",
                provider=self.cloud_provider,
                model_id=self.openrouter_large if self.openrouter_key else self.gemini_large,
                location="cloud",
                tier=2,
                cutoff=date(2025, 1, 1),
                price_in=_float("PRICE_LARGE_IN", 0.30),
                price_out=_float("PRICE_LARGE_OUT", 2.50),
            ),
        ]


def load_settings() -> Settings:
    keys = [_env(f"GEMINI_API_KEY_{i}") for i in range(1, 5)] + [_env("GEMINI_API_KEY")]
    return Settings(
        gemini_keys=[k for k in dict.fromkeys(keys) if k],
        gemini_light=_env("GEMINI_LIGHT_MODEL", "gemini-2.5-flash-lite"),
        gemini_large=_env("GEMINI_LARGE_MODEL", "gemini-2.5-flash"),
        openrouter_key=_env("OPENROUTER_API_KEY"),
        openrouter_light=_env("OPENROUTER_LIGHT_MODEL", "google/gemini-2.5-flash-lite"),
        openrouter_large=_env("OPENROUTER_LARGE_MODEL", "google/gemini-2.5-flash"),
        openrouter_embed=_env("OPENROUTER_EMBED_MODEL", "google/gemini-embedding-001"),
        ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2:3b"),
        embed_provider=_env("EMBED_PROVIDER", "gemini"),
        embed_model=_env("EMBED_MODEL", "nomic-embed-text"),
        tavily_key=_env("TAVILY_API_KEY"),
        database_url=_env("DATABASE_URL"),
        confidence_threshold=_float("CONFIDENCE_THRESHOLD", 0.6),
        retrieval_threshold=_float("RETRIEVAL_THRESHOLD", 0.55),
    )
