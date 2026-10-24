"""Cloud models through the Gemini REST API (T-09).

Rotates across every configured key and backs off on HTTP 429 / 5xx.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time

import httpx

from tempo.config import ModelSpec
from tempo.executors.base import Completion, ProviderError, estimate_tokens

BASE = "https://generativelanguage.googleapis.com/v1beta"


class KeyPool:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("no Gemini API keys configured")
        self.keys = keys
        self._cycle = itertools.cycle(keys)

    def next(self) -> str:
        return next(self._cycle)


class GeminiClient:
    def __init__(
        self,
        spec: ModelSpec,
        keys: KeyPool,
        retries: int = 4,
        timeout: float = 60.0,
        thinking_budget: int | None = None,
    ):
        self.spec = spec
        self.keys = keys
        self.retries = retries
        self.thinking_budget = thinking_budget
        self._http = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 512
    ) -> Completion:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if self.thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
            body["generationConfig"]["maxOutputTokens"] = max_tokens + max(0, self.thinking_budget)
        url = f"{BASE}/models/{self.spec.model_id}:generateContent"
        start = time.perf_counter()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = await self._http.post(
                    url, json=body, headers={"x-goog-api-key": self.keys.next()}
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                return self._parse(r.json(), prompt, start)
            except httpx.HTTPStatusError as e:
                last = e
                code = e.response.status_code
                if code == 400 and "thinking" in e.response.text.lower() and self.thinking_budget:
                    # Model does not accept a thinking budget: drop it and retry.
                    self.thinking_budget = None
                    body["generationConfig"].pop("thinkingConfig", None)
                    body["generationConfig"]["maxOutputTokens"] = max_tokens
                    continue
                if code not in (429, 500, 502, 503, 504):
                    raise ProviderError(f"gemini {code}: {e.response.text[:300]}") from e
            except httpx.HTTPError as e:
                last = e
            await asyncio.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.3)
        raise ProviderError(f"gemini: gave up after {self.retries} tries: {last}")

    def _parse(self, data: dict, prompt: str, start: float) -> Completion:
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        out_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
        return Completion(
            text=text,
            input_tokens=usage.get("promptTokenCount") or estimate_tokens(prompt),
            output_tokens=out_tokens or estimate_tokens(text),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def available(self) -> bool:
        return bool(self.keys.keys)
