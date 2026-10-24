"""Local model through Ollama (T-08). Cost is zero; data never leaves the machine."""

from __future__ import annotations

import time

import httpx

from tempo.config import ModelSpec
from tempo.executors.base import Completion, ProviderError, estimate_tokens


class OllamaClient:
    def __init__(self, spec: ModelSpec, host: str, timeout: float = 120.0):
        self.spec = spec
        self.host = host.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 512
    ) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": self.spec.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        start = time.perf_counter()
        try:
            r = await self._http.post(f"{self.host}/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama: {e}") from e
        data = r.json()
        text = data.get("message", {}).get("content", "")
        return Completion(
            text=text,
            input_tokens=data.get("prompt_eval_count") or estimate_tokens(prompt),
            output_tokens=data.get("eval_count") or estimate_tokens(text),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async def available(self) -> bool:
        try:
            r = await self._http.get(f"{self.host}/api/tags", timeout=2.0)
            names = {m.get("name", "") for m in r.json().get("models", [])}
            want = self.spec.model_id
            return any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
        except Exception:
            return False
