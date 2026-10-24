from fastapi.testclient import TestClient

from tempo import __version__
from tempo.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_ask_returns_result_with_receipt():
    r = client.post("/ask", json={"text": "What is the capital of France?", "deadline_ms": 2000})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"answer", "confidence", "latency_ms", "cost", "plan", "escalated"}
    assert body["plan"]["reason"], "every answer must carry a routing reason"


def test_private_request_stays_local():
    payload = {"text": "My email is a@b.com, summarise my notes", "private": True}
    r = client.post("/ask", json=payload)
    assert r.status_code == 200
    assert r.json()["plan"]["location"] == "local"


def test_run_plan_uses_forced_plan():
    plan = {"model": "gemini-large", "search": True, "location": "cloud", "reason": "baseline"}
    r = client.post("/run_plan", json={"req": {"text": "latest news"}, "plan": plan})
    assert r.status_code == 200
    assert r.json()["plan"]["model"] == "gemini-large"


def test_rejects_empty_text():
    r = client.post("/ask", json={"text": ""})
    assert r.status_code == 422
