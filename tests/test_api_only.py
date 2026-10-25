"""API-only mode: no Ollama, Gemini embeddings, personal data redacted before cloud calls."""

from tempo.core.engine import Tempo
from tempo.core.factory import build_clients, load_memory_docs
from tempo.core.schemas import Plan, Request
from tempo.executors.fake import FakeClient
from tempo.telemetry.embeddings import HashEmbedder
from tempo.telemetry.sink import NullTelemetry
from tempo.telemetry.vectors import MemoryVectorStore
from tests.conftest import FakeSearch, settings
from tests.test_router import plan_for

PII_QUESTION = "My email is omar.haddad@example.com. How many days are there in a leap year?"


class CloudEmbedder(HashEmbedder):
    """Hash embedder that pretends to be a cloud API and records what it was sent."""

    location = "cloud"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen += texts
        return await super().embed(texts)


async def api_only_engine(embedder=None, **answers) -> Tempo:
    s = settings(ollama_model="")
    specs = {m.name: m for m in s.models}
    clients = {
        name: FakeClient(specs[name], answers.get(name, "366 days")) for name in ("light", "large")
    }
    store = MemoryVectorStore(embedder or HashEmbedder(s.embed_dim))
    await load_memory_docs(store, s)
    engine = Tempo(s, clients, FakeSearch(), store, NullTelemetry())
    await engine.refresh()
    return engine


def test_blank_ollama_model_skips_the_local_client():
    assert "local" not in build_clients(settings(ollama_model=""))
    assert "local" in build_clients(settings(ollama_model="llama3.2:3b"))


def test_embeddings_default_to_gemini():
    assert settings().embed_provider == "gemini"


def test_planner_without_local_model_explains_redaction():
    p = plan_for(PII_QUESTION, available={"light", "large"})
    assert p.location == "cloud"
    assert "redacted" in p.reason


async def test_personal_data_is_redacted_before_the_cloud_model():
    engine = await api_only_engine()
    r = await engine.ask(Request(text=PII_QUESTION))
    sent = engine.clients["light"].calls
    assert sent and all("omar.haddad@example.com" not in p for p in sent)
    assert all("[email]" in p for p in sent)
    assert "redacted" in r.attempts[0].note
    assert r.answer == "366 days"


async def test_forced_cloud_plan_also_redacts():
    engine = await api_only_engine()
    await engine.run_fixed(Request(text=PII_QUESTION), Plan(model="light", location="cloud"))
    assert "omar.haddad@example.com" not in engine.clients["light"].calls[0]


async def test_requests_without_personal_data_are_sent_unchanged():
    engine = await api_only_engine()
    await engine.ask(Request(text="What is the capital of Australia?"))
    assert "What is the capital of Australia?" in engine.clients["light"].calls[0]


async def test_cloud_embedder_only_sees_redacted_queries():
    emb = CloudEmbedder()
    engine = await api_only_engine(embedder=emb)
    emb.seen.clear()  # drop the document chunks indexed at startup
    await engine.ask(Request(text=PII_QUESTION))
    assert emb.seen and all("omar.haddad@example.com" not in t for t in emb.seen)


async def test_no_models_at_all_still_answers_with_a_receipt():
    engine = Tempo(settings(ollama_model="", gemini_keys=[]), {})
    await engine.refresh()
    r = await engine.ask(Request(text="What is the capital of Australia?"))
    assert r.answer == ""
    assert "no model reachable" in r.receipt[0]
