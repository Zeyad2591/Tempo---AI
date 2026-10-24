"""Model client interface shared by every executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tempo.config import ModelSpec


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    def cost(self, spec: ModelSpec) -> float:
        return (self.input_tokens * spec.price_in + self.output_tokens * spec.price_out) / 1e6


class ModelClient(Protocol):
    spec: ModelSpec

    async def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2, max_tokens: int = 512
    ) -> Completion: ...

    async def available(self) -> bool: ...


class ProviderError(RuntimeError):
    """A provider call failed after retries."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
