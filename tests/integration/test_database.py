"""Checks the Postgres stack Tempo relies on: pgvector, pgvectorscale and TimescaleDB.

CI runs these against the timescaledb-ha image, which ships the same extensions as Tiger Cloud.
Locally they are skipped unless DATABASE_URL is set.
"""

import os

import pytest

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.fixture
def conn():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(DATABASE_URL, autocommit=True) as c:
        yield c


def test_vector_search_with_diskann(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE")
    conn.execute("DROP TABLE IF EXISTS ci_documents")
    conn.execute(
        "CREATE TABLE ci_documents (id serial PRIMARY KEY, body text, embedding vector(3))"
    )
    conn.execute(
        "INSERT INTO ci_documents (body, embedding) VALUES "
        "('refund policy', '[1,0,0]'), ('holiday rota', '[0,1,0]'), ('pricing', '[0,0,1]')"
    )
    conn.execute("CREATE INDEX ON ci_documents USING diskann (embedding vector_cosine_ops)")
    row = conn.execute(
        "SELECT body FROM ci_documents ORDER BY embedding <=> '[0.9,0.1,0]' LIMIT 1"
    ).fetchone()
    assert row[0] == "refund policy"


def test_telemetry_hypertable(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    conn.execute("DROP TABLE IF EXISTS ci_telemetry")
    conn.execute(
        "CREATE TABLE ci_telemetry (ts timestamptz NOT NULL, prompt_id text, plan jsonb, "
        "latency_ms int, cost double precision, correct boolean)"
    )
    conn.execute("SELECT create_hypertable('ci_telemetry', 'ts')")
    conn.execute(
        "INSERT INTO ci_telemetry VALUES (now(), 'p1', '{\"model\": \"local\"}', 120, 0, true)"
    )
    n = conn.execute("SELECT count(*) FROM ci_telemetry").fetchone()[0]
    assert n == 1
