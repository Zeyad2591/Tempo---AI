"""Web search (T-10). Tavily if a key is set; otherwise search is unavailable."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class SearchHit:
    title: str
    url: str
    content: str


class WebSearch:
    async def search(self, query: str, k: int = 4) -> list[SearchHit]:
        raise NotImplementedError

    async def available(self) -> bool:
        return False


class TavilySearch(WebSearch):
    def __init__(self, api_key: str, timeout: float = 15.0):
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=timeout)
        self.last_latency_ms = 0

    async def search(self, query: str, k: int = 4) -> list[SearchHit]:
        start = time.perf_counter()
        r = await self._http.post(
            "https://api.tavily.com/search",
            json={"query": query[:400], "max_results": k, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        self.last_latency_ms = int((time.perf_counter() - start) * 1000)
        return [
            SearchHit(h.get("title", ""), h.get("url", ""), h.get("content", "")[:1200])
            for h in r.json().get("results", [])[:k]
        ]

    async def available(self) -> bool:
        return bool(self.api_key)


class NoSearch(WebSearch):
    async def search(self, query: str, k: int = 4) -> list[SearchHit]:
        return []
