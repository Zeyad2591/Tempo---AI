"""Signal extractors (T-21, T-22). Cheap regex and keyword checks, well under 20 ms."""

from __future__ import annotations

import re
from datetime import date

from tempo.core.schemas import Signals

# ---------- privacy ----------

_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "uk_phone": re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b"),
    "ni_number": re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    "nhs_number": re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b"),
    "date_of_birth": re.compile(r"\b(date of birth|d\.o\.b|born on)\b", re.I),
    "self_name": re.compile(r"\bmy name is [A-Z][a-z]+", re.I),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"),
}
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str) -> list[str]:
    found = [name for name, rx in _PII_PATTERNS.items() if rx.search(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card_number")
            break
    return found


def redact(text: str) -> str:
    """Strip personal data before anything leaves the machine (e.g. a web search query)."""
    out = _CARD.sub("[card]", text)
    for name, rx in _PII_PATTERNS.items():
        if name in ("date_of_birth",):
            continue
        out = rx.sub(f"[{name}]", out)
    return out


# ---------- freshness ----------

_FRESH_WORDS = re.compile(
    r"\b(latest|current(ly)?|today|tonight|yesterday|this (week|month|year)|right now|"
    r"recent(ly)?|news|breaking|as of|newest|just (announced|released)|upcoming|"
    r"stock price|exchange rate|weather|score)\b",
    re.I,
)
_MONTHS = (
    "january february march april may june july august september october november december"
).split()
_MONTH_YEAR = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def freshness(text: str, cutoff: date) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    for month, year in _MONTH_YEAR.findall(text):
        d = date(int(year), _MONTHS.index(month.lower()) + 1, 1)
        if d > cutoff:
            score = max(score, 0.9)
            reasons.append(f"mentions {month.title()} {year}, after the model's cutoff")
    for y in {int(y) for y in _YEAR.findall(text)}:
        if date(y, 7, 1) > cutoff and not any(str(y) in r for r in reasons):
            score = max(score, 0.9)
            reasons.append(f"mentions {y}, after the model's cutoff")
    words = {w[0].lower() for w in _FRESH_WORDS.findall(text)}
    if words:
        score = max(score, min(1.0, 0.6 + 0.1 * (len(words) - 1)))
        reasons.append("time-sensitive words: " + ", ".join(sorted(words)))
    return score, reasons


# ---------- difficulty ----------

_STRONG_MATH = re.compile(
    r"\b(prime numbers?|primes|sum of|area|perimeter|equation|solve|derivative|integral|"
    r"probability|average speed|percentage increase|compound interest)\b",
    re.I,
)
_WORD_PROBLEM = re.compile(
    r"\b(each|total|save|average|twice|times|in \d+ years|how many (minutes|hours|days|weeks)|"
    r"per (hour|day|week)|remaining|left over|altogether)\b",
    re.I,
)
_CODE = re.compile(
    r"\b(python|javascript|sql|regex|function|bug|stack trace|compile|algorithm|big-o)\b|```", re.I
)
_REASONING = re.compile(
    r"\b(why|explain|prove|compare|derive|step by step|analy[sz]e|trade-?offs?|evaluate|justify)\b",
    re.I,
)
_MULTI_STEP = re.compile(r"\b(compare|and then|plan|steps?|first .* then)\b", re.I)
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)?")


def difficulty(text: str) -> tuple[float, bool, bool]:
    """Returns (difficulty 0..1, math_or_code, multi_step)."""
    nums = len(_NUMBER.findall(text))
    words = len(text.split())
    d = 0.1
    math = bool(_STRONG_MATH.search(text)) or (nums >= 2 and bool(_WORD_PROBLEM.search(text)))
    code = bool(_CODE.search(text))
    if math:
        d += 0.45
    if nums >= 3:
        d += 0.2
    if code:
        d += 0.4
    if _REASONING.search(text):
        d += 0.25
    if words > 40:
        d += 0.15
    if words > 80:
        d += 0.1
    multi = bool(_MULTI_STEP.search(text)) or text.count("?") > 1
    return min(1.0, d), math or code, multi


# ---------- private documents ----------

_DOMAIN_HINT = re.compile(
    r"\b(our|we|company|internal|policy|halden|fleet console|ardent|on-?call|rota|office|"
    r"expenses?|onboarding|sla|roadmap|postmortem|incident|leave)\b",
    re.I,
)


def domain_hint(text: str) -> bool:
    return bool(_DOMAIN_HINT.search(text))


def extract(text: str, cutoff: date, domain_score: float = 0.0) -> Signals:
    fresh, fresh_reasons = freshness(text, cutoff)
    diff, math, multi = difficulty(text)
    return Signals(
        difficulty=round(diff, 2),
        math_or_code=math,
        freshness=round(fresh, 2),
        fresh_reasons=fresh_reasons,
        pii=detect_pii(text),
        domain_score=round(domain_score, 3),
        domain_hint=domain_hint(text),
        multi_step=multi,
    )
