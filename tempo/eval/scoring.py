"""Scoring (T-32): is the answer correct, and did the router pick the expected route?"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from tempo.core.schemas import Plan
from tempo.verifier.verifier import final_number


@dataclass
class Item:
    id: str
    category: str
    prompt: str
    answers: list[str]
    numeric: float | None = None
    expect: dict = field(default_factory=dict)


def load_testset(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(Item(**json.loads(line)))
    return items


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9.% ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_correct(answer: str, item: Item) -> bool:
    if not answer.strip():
        return False
    a = f" {norm(answer)} "
    for alias in item.answers:
        n = norm(alias)
        if n and re.search(rf"(?<![a-z0-9.]){re.escape(n)}(?![a-z0-9]|\.\d)", a):
            return True
    if item.numeric is not None:
        got = final_number(answer)
        if got is not None and abs(got - item.numeric) <= 1e-6 * max(1.0, abs(item.numeric)):
            return True
    return False


def route_ok(plan: Plan, item: Item) -> bool | None:
    """True/False against item.expect, None if the item has no routing expectation."""
    if not item.expect:
        return None
    for key, want in item.expect.items():
        if getattr(plan, key, None) != want:
            return False
    return True
