"""Embedders. Ollama keeps private documents and queries on the machine."""

from __future__ import annotations

import hashlib
import math
import re

import httpx

from tempo.executors.gemini import BASE, KeyPool


def normalise(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Deterministic bag-of-words hashing. Offline fallback and for tests."""

    location = "local"

    def __init__(self, dim: int = 768):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if len(w) < 3:
                    continue
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            out.append(normalise(v))
        return out


class OllamaEmbedder:
    location = "local"

    def __init__(self, host: str, model: str = "nomic-embed-text", dim: int = 768):
        self.host = host.rstrip("/")
        self.model = model
        self.dim = dim
        self._http = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await self._http.post(
            f"{self.host}/api/embed", json={"model": self.model, "input": texts}
        )
        r.raise_for_status()
        return [normalise(v[: self.dim]) for v in r.json()["embeddings"]]


class OpenRouterEmbedder:
    location = "cloud"  # queries are redacted before they are embedded

    def __init__(self, api_key: str, model: str = "google/gemini-embedding-001", dim: int = 768):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self._http = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await self._http.post(
            "https://openrouter.ai/api/v1/embeddings",
            json={"model": self.model, "input": texts, "dimensions": self.dim},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        # Truncating keeps the schema's 768 dims if the provider ignores `dimensions`.
        return [normalise(d["embedding"][: self.dim]) for d in rows]


class GeminiEmbedder:
    location = "cloud"  # queries are redacted before they are embedded

    def __init__(self, keys: KeyPool, model: str = "gemini-embedding-001", dim: int = 768):
        self.keys = keys
        self.model = model
        self.dim = dim
        self._http = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        reqs = [
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": self.dim,
            }
            for t in texts
        ]
        r = await self._http.post(
            f"{BASE}/models/{self.model}:batchEmbedContents",
            json={"requests": reqs},
            headers={"x-goog-api-key": self.keys.next()},
        )
        r.raise_for_status()
        return [normalise(e["values"]) for e in r.json()["embeddings"]]
