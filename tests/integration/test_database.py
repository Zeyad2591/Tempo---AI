"""Tempo's Tiger Data layer against a real Postgres with pgvector, pgvectorscale and TimescaleDB.

CI applies db/schema.sql to a timescaledb-ha service first (same extensions as Tiger Cloud).
Locally these are skipped unless DATABASE_URL is set.
"""

import os

import pytest

from tempo.core.schemas import Plan, TelemetryRow
from tempo.telemetry.embeddings import HashEmbedder
from tempo.telemetry.sink import TigerTelemetry
from tempo.telemetry.vectors import TigerVectorStore

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.fixture
def url() -> str:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    return DATABASE_URL


async def test_vector_store_round_trip(url):
    store = TigerVectorStore(url, HashEmbedder(768))
    await store.clear()
    await store.add(
        [
            ("refund-policy", "Customers can request a full refund within 45 days of delivery."),
            ("annual-leave", "Every full-time employee gets 27 days of annual leave."),
        ]
    )
    assert await store.count() == 2
    hits = await store.search("how many days of annual leave does an employee get", k=1)
    assert hits[0].doc == "annual-leave"
    assert 0 < hits[0].score <= 1


async def test_telemetry_hypertable_insert(url):
    import psycopg

    sink = TigerTelemetry(url)
    await sink.log(
        TelemetryRow(
            request_id="ci",
            prompt_id="f01",
            plan=Plan(model="local", location="local"),
            latency_ms=120,
            cost=0.0,
            confidence=0.9,
        )
    )
    with psycopg.connect(url) as c:
        n = c.execute("SELECT count(*) FROM telemetry WHERE request_id = 'ci'").fetchone()[0]
        hyper = c.execute(
            "SELECT count(*) FROM timescaledb_information.hypertables "
            "WHERE hypertable_name = 'telemetry'"
        ).fetchone()[0]
    assert n >= 1 and hyper == 1
