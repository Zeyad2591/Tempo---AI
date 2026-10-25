"""OpenRouter client and embedder, against a mocked HTTP transport. No network."""

import json

import httpx
import pytest

from tempo.core.factory import build_clients, build_embedder
from tempo.executors.base import ProviderError
from tempo.executors.openrouter import OpenRouterClient
from tempo.telemetry.embeddings import OpenRouterEmbedder
from tests.conftest import settings

OK = {
    "choices": [{"message": {"role": "assistant", "content": "Canberra"}}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 3},
}


def or_settings(**kw):
    return settings(openrouter_key="sk-or-test", gemini_keys=[], ollama_model="", **kw)


def mocked(client, handler):
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def light_client(handler, **kw) -> OpenRouterClient:
    spec = {m.name: m for m in or_settings().models}["light"]
    return mocked(OpenRouterClient(spec, "sk-or-test", **kw), handler)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("tempo.executors.openrouter.asyncio.sleep", instant)


def test_openrouter_key_routes_cloud_tiers_through_openrouter():
    clients = build_clients(or_settings())
    assert set(clients) == {"light", "large"}
    assert all(isinstance(c, OpenRouterClient) for c in clients.values())
    assert clients["light"].spec.model_id == "google/gemini-2.5-flash-lite"
    assert clients["large"].spec.provider == "openrouter"


def test_only_openrouter_key_embeds_through_openrouter():
    assert isinstance(build_embedder(or_settings()), OpenRouterEmbedder)


async def test_generate_sends_chat_request_and_parses_usage():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=OK)

    c = await light_client(handler, thinking_budget=0).generate("Capital?", system="Be brief")
    assert c.text == "Canberra" and c.input_tokens == 20 and c.output_tokens == 3
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["body"]["model"] == "google/gemini-2.5-flash-lite"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief"}
    assert seen["body"]["reasoning"] == {"enabled": False}


async def test_retries_on_rate_limit_then_succeeds():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return (
            httpx.Response(429, text="slow down")
            if len(calls) == 1
            else httpx.Response(200, json=OK)
        )

    assert (await light_client(handler).generate("Capital?")).text == "Canberra"
    assert len(calls) == 2


async def test_error_inside_a_200_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 400, "message": "bad model"}})

    with pytest.raises(ProviderError, match="bad model"):
        await light_client(handler).generate("Capital?")


async def test_bad_key_is_not_retried():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="invalid key")

    with pytest.raises(ProviderError, match="401"):
        await light_client(handler).generate("Capital?")
    assert len(calls) == 1


async def test_embedder_keeps_input_order_and_768_dims():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["model"] == "google/gemini-embedding-001" and body["dimensions"] == 768
        data = [
            {"index": 1, "embedding": [0.0, 2.0] * 1000},
            {"index": 0, "embedding": [3.0] * 2000},
        ]
        return httpx.Response(200, json={"data": data})

    emb = mocked(OpenRouterEmbedder("sk-or-test"), handler)
    first, second = await emb.embed(["a", "b"])
    assert len(first) == len(second) == 768
    assert first[1] > 0 and second[0] == 0.0
