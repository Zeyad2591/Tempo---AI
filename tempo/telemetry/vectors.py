"""Vector store for private documents (T-11, T-28, T-29).

TigerVectorStore uses pgvector + pgvectorscale (DiskANN) on Tiger Cloud.
MemoryVectorStore is the offline fallback with the same interface.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Protocol


class Embedder(Protocol):
    dim: int
    location: str  # "local" | "cloud"

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class Passage:
    doc: str
    text: str
    score: float  # cosine similarity, 1 = identical
    meta: dict = field(default_factory=dict)


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


class VectorStore:
    embedder: Embedder

    async def add(self, items: list[tuple[str, str]]) -> int:
        """items = [(doc_name, chunk_text)]"""
        raise NotImplementedError

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        raise NotImplementedError

    async def count(self) -> int:
        raise NotImplementedError


class MemoryVectorStore(VectorStore):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.rows: list[tuple[str, str, list[float]]] = []

    async def add(self, items: list[tuple[str, str]]) -> int:
        vecs = await self.embedder.embed([t for _, t in items])
        self.rows += [(d, t, v) for (d, t), v in zip(items, vecs, strict=True)]
        return len(items)

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        if not self.rows:
            return []
        (q,) = await self.embedder.embed([query])
        scored = sorted(((_cos(q, v), d, t) for d, t, v in self.rows), reverse=True)[:k]
        return [Passage(d, t, s) for s, d, t in scored]

    async def count(self) -> int:
        return len(self.rows)


class TigerVectorStore(VectorStore):
    """Expects db/schema.sql to have been applied."""

    def __init__(self, database_url: str, embedder: Embedder):
        self.url = database_url
        self.embedder = embedder

    def _connect(self):
        import psycopg

        return psycopg.connect(self.url, autocommit=True)

    async def add(self, items: list[tuple[str, str]]) -> int:
        vecs = await self.embedder.embed([t for _, t in items])

        def write() -> None:
            with self._connect() as c, c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO documents (doc, body, embedding) VALUES (%s, %s, %s::vector)",
                    [(d, t, json.dumps(v)) for (d, t), v in zip(items, vecs, strict=True)],
                )

        await asyncio.to_thread(write)
        return len(items)

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        (q,) = await self.embedder.embed([query])

        def read() -> list[Passage]:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT doc, body, 1 - (embedding <=> %s::vector) AS score "
                    "FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
                    (json.dumps(q), json.dumps(q), k),
                ).fetchall()
            return [Passage(d, b, float(s)) for d, b, s in rows]

        return await asyncio.to_thread(read)

    async def count(self) -> int:
        def read() -> int:
            with self._connect() as c:
                return c.execute("SELECT count(*) FROM documents").fetchone()[0]

        return await asyncio.to_thread(read)

    async def clear(self) -> None:
        def run() -> None:
            with self._connect() as c:
                c.execute("TRUNCATE documents")

        await asyncio.to_thread(run)


def chunk(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Split on paragraphs, packing them up to ~size characters with a little overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks
