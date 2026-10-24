import pytest
from fastapi.testclient import TestClient

from tempo import __version__
from tempo.api.main import app
from tests.conftest import make_engine


@pytest.fixture
async def client():
    app.state.engine = await make_engine()
    with TestClient(app) as c:
        yield c
    app.state.engine = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["models"] == {"local": True, "light": True, "large": True}


def test_ask_returns_result_with_receipt(client):
    r = client.post("/ask", json={"text": "What is the capital of France?", "deadline_ms": 2000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"answer", "confidence", "latency_ms", "cost", "plan", "receipt"}
    assert body["plan"]["reason"], "every answer must carry a routing reason"


def test_private_request_stays_local(client):
    payload = {"text": "My email is a@b.com, summarise my notes", "private": True}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200
    assert r.json()["plan"]["location"] == "local"


def test_run_plan_uses_forced_plan(client):
    plan = {"model": "large", "location": "cloud", "reason": "baseline"}
    r = client.post("/run_plan", json={"req": {"text": "latest news"}, "plan": plan})
    assert r.status_code == 200
    assert r.json()["plan"]["model"] == "large"


def test_rejects_empty_text(client):
    assert client.post("/ask", json={"text": ""}).status_code == 422
