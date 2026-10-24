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
    deadline_ms: int = Field(default=5000, gt=0)
    budget: float = Field(default=0.01, ge=0, description="Max cost in USD for this request")
    private: bool = False
    accuracy_target: float = Field(default=0.9, ge=0, le=1)
    history: list[Message] = Field(default_factory=list)


class Plan(BaseModel):
    """The execution plan the router picks: pi = (model, search, retrieve, agent, location)."""

    model: str
    search: bool = False
    retrieve: bool = False
    agent: bool = False
    location: Literal["local", "cloud"] = "local"
    reason: str = ""


class Result(BaseModel):
    """What Tempo returns, including the routing receipt (the plan and its reason)."""

    answer: str
    confidence: float = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)
    cost: float = Field(ge=0)
    plan: Plan
    escalated: bool = False
    sources: list[str] = Field(default_factory=list)


class TelemetryRow(BaseModel):
    """One row in the Tiger Data telemetry hypertable."""

    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_id: str
    plan: Plan
    latency_ms: int = Field(ge=0)
    cost: float = Field(ge=0)
    correct: bool | None = None
