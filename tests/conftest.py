"""Shared test helpers: an engine wired to scripted fake models. No network."""

from __future__ import annotations

from pathlib import Path

import pytest

from tempo.config import Settings
from tempo.core.engine import Tempo
from tempo.core.factory import load_memory_docs
from tempo.executors.fake import FakeClient
from tempo.executors.search import SearchHit, WebSearch
from tempo.telemetry.embeddings import HashEmbedder
from tempo.telemetry.sink import NullTelemetry
from tempo.telemetry.vectors import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]


class FakeSearch(WebSearch):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 4) -> list[SearchHit]:
        self.queries.append(query)
        return [SearchHit("Pope news", "https://example.com/pope", "Pope Leo XIV was elected.")]

    async def available(self) -> bool:
        return True


def settings(**kw) -> Settings:
    base = dict(gemini_keys=["test"], data_dir=ROOT / "data", retrieval_threshold=0.3)
    base.update(kw)
    return Settings(**base)


async def make_engine(answers: dict[str, object] | None = None, search=True, **kw) -> Tempo:
    s = settings(**kw)
    specs = {m.name: m for m in s.models}
    answers = answers or {}
    clients = {
        name: FakeClient(specs[name], answers.get(name, f"{name} says Canberra"))
        for name in ("local", "light", "large")
    }
    store = MemoryVectorStore(HashEmbedder(s.embed_dim))
    await load_memory_docs(store, s)
    engine = Tempo(s, clients, FakeSearch() if search else None, store, NullTelemetry())
    await engine.refresh()
    return engine


@pytest.fixture
async def engine() -> Tempo:
    return await make_engine()
