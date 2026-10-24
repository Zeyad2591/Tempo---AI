-- Tempo schema for Tiger Cloud (T-28). Safe to run more than once.
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;  -- also installs pgvector
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Private documents for retrieval. 768 dims matches nomic-embed-text and
-- gemini-embedding-001 with outputDimensionality=768.
CREATE TABLE IF NOT EXISTS documents (
    id         bigserial PRIMARY KEY,
    doc        text NOT NULL,
    body       text NOT NULL,
    embedding  vector(768) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING diskann (embedding vector_cosine_ops);

-- One row per attempt (an escalated request logs every attempt).
CREATE TABLE IF NOT EXISTS telemetry (
    ts          timestamptz NOT NULL,
    request_id  text,
    prompt_id   text,
    system      text,          -- tempo | always_largest | always_cheapest | always_retrieve
    attempt     int,
    model       text,
    search      boolean,
    retrieve    boolean,
    location    text,
    latency_ms  int,
    cost        double precision,
    confidence  double precision,
    escalated   boolean,
    correct     boolean,
    plan        jsonb
);
SELECT create_hypertable('telemetry', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS telemetry_model_ts_idx ON telemetry (model, ts DESC);
