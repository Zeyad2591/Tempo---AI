"""Builds a ready engine from settings, falling back gracefully when a service is missing."""

from __future__ import annotations

import logging
import os

from tempo.config import Settings, load_settings
from tempo.core.engine import Tempo
from tempo.executors.base import ModelClient
from tempo.executors.gemini import GeminiClient, KeyPool
from tempo.executors.ollama import OllamaClient
from tempo.executors.openrouter import OpenRouterClient
from tempo.executors.search import NoSearch, TavilySearch, WebSearch
from tempo.telemetry.embeddings import (
    GeminiEmbedder,
    HashEmbedder,
    OllamaEmbedder,
    OpenRouterEmbedder,
)
from tempo.telemetry.sink import JsonlTelemetry, TelemetrySink, TigerTelemetry
from tempo.telemetry.vectors import (
    Embedder,
    MemoryVectorStore,
    TigerVectorStore,
    VectorStore,
    chunk,
)

log = logging.getLogger("tempo")


def build_clients(s: Settings) -> dict[str, ModelClient]:
    specs = {m.name: m for m in s.models}
    clients: dict[str, ModelClient] = {}
    if s.ollama_model:  # blank OLLAMA_MODEL = API-only, never contact Ollama
        clients["local"] = OllamaClient(specs["local"], s.ollama_host)
    raw = os.environ.get("LARGE_THINKING_BUDGET", "512").strip().lower()
    large_budget = None if raw in ("", "none") else int(raw)
    if s.openrouter_key:  # OpenRouter wins when set; see Settings.cloud_provider
        key = s.openrouter_key
        clients["light"] = OpenRouterClient(specs["light"], key, thinking_budget=0)
        clients["large"] = OpenRouterClient(specs["large"], key, thinking_budget=large_budget)
    elif s.gemini_keys:
        pool = KeyPool(s.gemini_keys)
        clients["light"] = GeminiClient(specs["light"], pool, thinking_budget=0)
        clients["large"] = GeminiClient(specs["large"], pool, thinking_budget=large_budget)
    return clients


def build_embedder(s: Settings) -> Embedder:
    # "gemini" with only an OpenRouter key: the same Gemini embedding model, via OpenRouter.
    wants_openrouter = s.embed_provider == "openrouter" or (
        s.embed_provider == "gemini" and not s.gemini_keys
    )
    if wants_openrouter and s.openrouter_key:
        return OpenRouterEmbedder(s.openrouter_key, s.openrouter_embed, dim=s.embed_dim)
    if s.embed_provider == "gemini" and s.gemini_keys:
        return GeminiEmbedder(KeyPool(s.gemini_keys), dim=s.embed_dim)
    if s.embed_provider == "ollama":
        return OllamaEmbedder(s.ollama_host, s.embed_model, s.embed_dim)
    return HashEmbedder(s.embed_dim)


def build_vector_store(s: Settings, embedder: Embedder) -> VectorStore:
    if s.database_url:
        return TigerVectorStore(s.database_url, embedder)
    return MemoryVectorStore(embedder)


def build_telemetry(s: Settings) -> TelemetrySink:
    if s.database_url:
        return TigerTelemetry(s.database_url)
    return JsonlTelemetry(s.results_dir / "telemetry.jsonl")


def build_search(s: Settings) -> WebSearch:
    return TavilySearch(s.tavily_key) if s.tavily_key else NoSearch()


async def _working_embedder(s: Settings) -> Embedder:
    emb = build_embedder(s)
    try:
        await emb.embed(["ping"])
        return emb
    except Exception as e:
        log.warning(
            "embedder %s unavailable (%s); using offline hash embedder", s.embed_provider, e
        )
        return HashEmbedder(s.embed_dim)


async def load_memory_docs(store: MemoryVectorStore, s: Settings) -> int:
    items = [
        (p.stem, c)
        for p in sorted((s.data_dir / "docs").glob("*.md"))
        for c in chunk(p.read_text(encoding="utf-8"))
    ]
    return await store.add(items) if items else 0


async def build_engine(s: Settings | None = None) -> Tempo:
    s = s or load_settings()
    embedder = await _working_embedder(s)
    store = build_vector_store(s, embedder)
    if isinstance(store, MemoryVectorStore):
        await load_memory_docs(store, s)
    engine = Tempo(s, build_clients(s), build_search(s), store, build_telemetry(s))
    await engine.refresh()
    log.info("tempo ready: %s", engine.status())
    return engine
