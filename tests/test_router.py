from datetime import date

from tempo.config import Settings
from tempo.core.schemas import Plan, Request, Signals
from tempo.router.rules import LatencyTracker, RulesPlanner, next_stronger
from tempo.router.signals import detect_pii, difficulty, extract, freshness, redact

CUTOFF = date(2025, 1, 1)
ALL = {"local", "light", "large"}


def planner() -> RulesPlanner:
    models = {m.name: m for m in Settings().models}
    return RulesPlanner(models, LatencyTracker(), retrieval_threshold=0.55)


def plan_for(text: str, domain_score: float = 0.0, available=ALL, **req) -> Plan:
    s = extract(text, CUTOFF, domain_score)
    return planner().plan(Request(text=text, **req), s, set(available), can_search=True)


# ---------- signals ----------


def test_pii_detection():
    assert "email" in detect_pii("mail me at omar.haddad@example.com")
    assert "uk_phone" in detect_pii("call 07700 900123")
    assert "card_number" in detect_pii("card 4111 1111 1111 1111")
    assert "card_number" not in detect_pii("order 1234 5678 9012 3456")  # fails Luhn
    assert "ni_number" in detect_pii("NI QQ 12 34 56 C")
    assert detect_pii("What is the capital of France?") == []


def test_redact_removes_pii():
    out = redact("I'm at a@b.com, card 4111 1111 1111 1111")
    assert "a@b.com" not in out and "4111" not in out


def test_freshness_after_cutoff():
    score, reasons = freshness("Who was elected Pope in May 2025?", CUTOFF)
    assert score >= 0.9 and reasons
    assert freshness("Who won the 2018 World Cup?", CUTOFF)[0] == 0
    assert freshness("What's the latest news on Mars?", CUTOFF)[0] >= 0.6


def test_difficulty_flags_word_problems():
    d, math, _ = difficulty("A train travels 180 km in 1.5 hours then 120 km. Average speed?")
    assert math and d >= 0.5
    d, math, _ = difficulty("What is the capital of Australia?")
    assert not math and d < 0.3


# ---------- rules ----------


def test_simple_fact_goes_local():
    p = plan_for("What is the capital of Australia?")
    assert p.model == "local" and not p.search and not p.retrieve
    assert "local model" in p.reason


def test_personal_data_stays_local_even_if_hard():
    p = plan_for("My email is a@b.com. What is the sum of all primes below 30?")
    assert p.model == "local" and p.location == "local"
    assert "personal data" in p.reason


def test_fresh_question_searches_with_cloud_model():
    p = plan_for("Who was elected Pope in May 2025?")
    assert p.search and p.model == "light"


def test_hard_maths_goes_large():
    p = plan_for("What is the sum of all prime numbers less than 30?")
    assert p.model == "large"


def test_private_docs_retrieve_locally_without_web_search():
    p = plan_for("Who is on call for the week starting 26 October 2026?", domain_score=0.7)
    assert p.retrieve and not p.search and p.model == "local"


def test_tight_deadline_drops_large():
    p = plan_for("What is the sum of all prime numbers less than 30?", deadline_ms=1500)
    assert p.model == "light"
    assert "deadline" in p.reason


def test_tiny_budget_downgrades():
    p = plan_for("What is the sum of all prime numbers less than 30?", budget=0.0)
    assert p.model in ("light", "local")
    assert "budget" in p.reason


def test_no_gemini_keys_everything_local():
    p = plan_for("What is the sum of all prime numbers less than 30?", available={"local"})
    assert p.model == "local"


def test_every_plan_has_a_reason():
    for text in ["hi there", "Explain why the sky is blue", "latest Bitcoin price"]:
        assert plan_for(text).reason


# ---------- escalation ladder ----------


def test_ladder_goes_up_then_adds_search():
    p = Plan(model="local")
    p1 = next_stronger(p, ALL, private=False, can_search=True)
    assert p1.model == "light"
    p2 = next_stronger(p1, ALL, private=False, can_search=True)
    assert p2.model == "large"
    p3 = next_stronger(p2, ALL, private=False, can_search=True)
    assert p3.model == "large" and p3.search
    assert next_stronger(p3, ALL, private=False, can_search=True) is None


def test_ladder_never_sends_private_data_to_cloud():
    p = next_stronger(Plan(model="local"), ALL, private=True, can_search=False)
    assert p is None


def test_prime_minister_is_not_maths():
    _, math, _ = difficulty("Who became Prime Minister of Canada in March 2025?")
    assert not math


def test_personal_data_is_never_web_searched():
    p = plan_for("My date of birth is 14 March 1998, I live at 22 Elm Road. Age in March 2027?")
    assert p.location == "local" and not p.search
    assert "never leaves" in p.reason


def test_signals_model_round_trip():
    s = extract("My phone is 07700 900123", CUTOFF)
    assert Signals.model_validate(s.model_dump()) == s
