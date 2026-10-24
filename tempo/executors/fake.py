"""Scripted model client for tests and offline demos. No network."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from tempo.config import ModelSpec
from tempo.executors.base import Completion, estimate_tokens


class FakeClient:
    def __init__(
        self,
        spec: ModelSpec,
        answer: str | Callable[[str, float], str] = "fake answer",
        latency_ms: int = 5,
        up: bool = True,
    ):
        self.spec = spec
        self.answer = answer
        self.latency_ms = latency_ms
        self.up = up
        self.calls: list[str] = []

    async def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 512
    ) -> Completion:
        self.calls.append(prompt)
        await asyncio.sleep(self.latency_ms / 1000)
        text = self.answer(prompt, temperature) if callable(self.answer) else self.answer
        return Completion(text, estimate_tokens(prompt), estimate_tokens(text), self.latency_ms)

    async def available(self) -> bool:
        return self.up
