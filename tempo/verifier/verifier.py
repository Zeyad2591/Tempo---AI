"""Verifier (T-14): confidence from sample agreement and grounding in the context."""

from __future__ import annotations

import re

_STOP = set(
    "a an the of to in on at for and or but is are was were be been it its this that these those "
    "with as by from into than then so if not no yes do does did can could would should will "
    "what which who whom whose when where why how i you he she we they them his her their our "
    "your my me us about there here also just only very more most answer final".split()
)
_REFUSAL = re.compile(
    r"\b(i (do not|don't|cannot|can't) (know|find|answer|access)|not (mentioned|provided|"
    r"specified|available) in|no information|unable to (answer|determine)|"
    r"as of my (last|knowledge))",
    re.I,
)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def final_number(text: str) -> float | None:
    m = re.search(r"final answer\s*[:\-]?\s*(.+)", text, re.I)
    scope = m.group(1) if m else text
    nums = _NUM.findall(scope)
    if not nums:
        return None
    raw = (nums[0] if m else nums[-1]).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def head(text: str, words: int = 60) -> str:
    return " ".join(text.split()[:words])


def agreement(a: str, b: str) -> float:
    """1.0 when two samples say the same thing, 0.0 when they clearly differ."""
    na, nb = final_number(a), final_number(b)
    if na is not None and nb is not None:
        return 1.0 if abs(na - nb) <= 1e-6 * max(1.0, abs(na)) else 0.0
    wa, wb = content_words(head(a)), content_words(head(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def grounding(answer: str, context: str) -> float:
    """Share of the answer's content words (and numbers) that appear in the context."""
    words = content_words(head(answer, 40))
    if not words:
        return 0.0
    ctx = content_words(context)
    return len(words & ctx) / len(words)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text)) or not text.strip()


def confidence(samples: list[str], context: str = "") -> tuple[float, str]:
    """Returns (confidence 0..1, explanation)."""
    first = samples[0] if samples else ""
    if is_refusal(first):
        return 0.15, "model could not answer"
    parts: list[str] = []
    if len(samples) >= 2:
        agree = agreement(samples[0], samples[1])
        parts.append(f"samples agree {agree:.2f}")
    else:
        agree = 0.8
        parts.append("single sample")
    if context:
        ground = grounding(first, context)
        parts.append(f"grounded {ground:.2f}")
        conf = 0.2 + 0.45 * agree + 0.35 * ground
    else:
        conf = 0.25 + 0.7 * agree
    return round(min(1.0, conf), 3), ", ".join(parts)
