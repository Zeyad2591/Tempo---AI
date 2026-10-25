"""Cloud models through OpenRouter's OpenAI-compatible API.

One key reaches many providers; by default the same Gemini models as the direct client.
Backs off on HTTP 429 / 5xx.
"""

from __future__ import annotations

import asyncio
import random
import time

import httpx

from tempo.config import ModelSpec
from tempo.executors.base import Completion, ProviderError, estimate_tokens

BASE = "https://openrouter.ai/api/v1"
HEADERS = {"X-Title": "Tempo"}
RETRY = (429, 500, 502, 503, 504)


class OpenRouterClient:
    def __init__(
        self,
        spec: ModelSpec,
        api_key: str,
        retries: int = 4,
        timeout: float = 60.0,
        thinking_budget: int | None = None,
    ):
        self.spec = spec
        self.api_key = api_key
        self.retries = retries
        self.thinking_budget = thinking_budget
        self._http = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 512
    ) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.thinking_budget is not None:
            if self.thinking_budget > 0:
                body["reasoning"] = {"max_tokens": self.thinking_budget, "exclude": True}
                body["max_tokens"] = max_tokens + self.thinking_budget
            else:
                body["reasoning"] = {"enabled": False}
        headers = {"Authorization": f"Bearer {self.api_key}", **HEADERS}
        start = time.perf_counter()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = await self._http.post(f"{BASE}/chat/completions", json=body, headers=headers)
                if r.status_code in RETRY:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                err = data.get("error")
                if err:  # OpenRouter can report upstream failures inside a 200
                    if err.get("code") in RETRY:
                        raise ProviderError(f"openrouter upstream {err.get('code')}")
                    raise ProviderError(f"openrouter: {str(err.get('message'))[:300]}")
                return self._parse(data, prompt, start)
            except httpx.HTTPStatusError as e:
                last = e
                code = e.response.status_code
                if code == 400 and "reasoning" in e.response.text.lower() and "reasoning" in body:
                    # Model does not accept reasoning settings: drop them and retry.
                    self.thinking_budget = None
                    body.pop("reasoning")
                    body["max_tokens"] = max_tokens
                    continue
                if code not in RETRY:
                    raise ProviderError(f"openrouter {code}: {e.response.text[:300]}") from e
            except ProviderError as e:
                if "upstream" not in str(e):
                    raise
                last = e
            except httpx.HTTPError as e:
                last = e
            await asyncio.sleep(min(8.0, 0.5 * 2**attempt) + random.random() * 0.3)
        raise ProviderError(f"openrouter: gave up after {self.retries} tries: {last}")

    def _parse(self, data: dict, prompt: str, start: float) -> Completion:
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens") or estimate_tokens(prompt),
            output_tokens=usage.get("completion_tokens") or estimate_tokens(text),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def available(self) -> bool:
        return bool(self.api_key)
