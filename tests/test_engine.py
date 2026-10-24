from tempo.core.schemas import Plan, Request
from tests.conftest import make_engine


async def test_simple_question_answered_locally(engine):
    r = await engine.ask(Request(text="What is the capital of Australia?"))
    assert r.plan.model == "local"
    assert not r.escalated
    assert r.cost == 0
    assert r.receipt and "local" in r.receipt[0]


async def test_disagreeing_samples_escalate_to_cloud():
    # The local model contradicts itself between samples, the light model is consistent.
    def flaky(prompt: str, temperature: float) -> str:
        return "Sydney" if temperature > 0.5 else "Melbourne"

    engine = await make_engine({"local": flaky, "light": "Canberra"})
    r = await engine.ask(Request(text="What is the capital of Australia?"))
    assert r.escalated
    assert r.plan.model == "light"
    assert r.answer == "Canberra"
    assert any("escalate" in line for line in r.receipt)
    assert len(r.attempts) == 2


async def test_private_data_never_escalates_to_cloud():
    def flaky(prompt: str, temperature: float) -> str:
        return "36" if temperature < 0.5 else "48"

    engine = await make_engine({"local": flaky})
    r = await engine.ask(Request(text="My email is a@b.com. What is 15% of 240?"))
    assert r.plan.location == "local"
    assert all(a.plan.location == "local" for a in r.attempts)
    assert engine.clients["light"].calls == [] and engine.clients["large"].calls == []


async def test_fresh_question_uses_redacted_search():
    engine = await make_engine({"light": "Pope Leo XIV"})
    r = await engine.ask(Request(text="Who was elected Pope in May 2025?"))
    assert r.plan.search
    assert "https://example.com/pope" in r.sources
    assert "Pope Leo XIV" in r.answer


async def test_private_docs_are_retrieved():
    engine = await make_engine({"local": "Customers have 45 days to request a full refund."})
    r = await engine.ask(
        Request(text="How many days do customers have to request a full refund for a Halden arm?")
    )
    assert r.plan.retrieve
    assert "refund-policy" in r.sources


async def test_run_fixed_forces_plan(engine):
    r = await engine.run_fixed(Request(text="hi"), Plan(model="large", location="cloud"))
    assert r.plan.model == "large"
    assert engine.clients["large"].calls


async def test_telemetry_logged_per_attempt(engine):
    await engine.ask(Request(text="What is the capital of Australia?", prompt_id="f01"))
    rows = engine.telemetry.rows
    assert rows and rows[0].prompt_id == "f01" and rows[0].system == "tempo"


async def test_missing_model_fails_soft():
    engine = await make_engine()
    engine.clients["local"].up = False
    await engine.refresh()
    r = await engine.ask(Request(text="What is the capital of Australia?"))
    assert r.plan.model != "local"
