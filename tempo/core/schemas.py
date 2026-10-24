"""Shared interfaces (T-02). Every module builds against these shapes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class Request(BaseModel):
    """What the caller sends to Tempo."""

    text: str = Field(min_length=1)
    deadline_ms: int = Field(default=8000, gt=0)
    budget: float = Field(default=0.01, ge=0, description="Max cost in USD for this request")
    private: bool = False
    accuracy_target: float = Field(default=0.9, ge=0, le=1)
    history: list[Message] = Field(default_factory=list)
    prompt_id: str | None = None


class Plan(BaseModel):
    """The execution plan the router picks: pi = (model, search, retrieve, agent, location)."""

    model: str  # "local" | "light" | "large" (names from config.Settings.models)
    search: bool = False
    retrieve: bool = False
    agent: bool = False
    location: Literal["local", "cloud"] = "local"
    reason: str = ""

    def label(self) -> str:
        extras = [x for x, on in (("search", self.search), ("retrieve", self.retrieve)) if on]
        return "+".join([self.model, *extras])


class Signals(BaseModel):
    """What the router measured about a request."""

    difficulty: float = 0.0
    math_or_code: bool = False
    freshness: float = 0.0
    fresh_reasons: list[str] = Field(default_factory=list)
    pii: list[str] = Field(default_factory=list)
    domain_score: float = 0.0
    domain_hint: bool = False
    multi_step: bool = False


class Attempt(BaseModel):
    plan: Plan
    answer: str
    confidence: float
    latency_ms: int
    cost: float
    note: str = ""


class Result(BaseModel):
    """What Tempo returns, including the routing receipt."""

    answer: str
    confidence: float = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)
    cost: float = Field(ge=0)
    plan: Plan
    escalated: bool = False
    sources: list[str] = Field(default_factory=list)
    signals: Signals | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    receipt: list[str] = Field(default_factory=list)


class TelemetryRow(BaseModel):
    """One row in the Tiger Data telemetry hypertable."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    request_id: str = ""
    prompt_id: str | None = None
    system: str = "tempo"
    attempt: int = 1
    plan: Plan
    latency_ms: int = Field(ge=0)
    cost: float = Field(ge=0)
    confidence: float | None = None
    escalated: bool = False
    correct: bool | None = None
