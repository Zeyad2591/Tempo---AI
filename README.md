# Tempo

A time-aware AI runtime that picks the fastest answer path that is still right, at the lowest cost.

For every request, Tempo decides which model to use (local or cloud, small or large), whether to search the web, and whether to retrieve from private documents. It plans around the request's deadline and budget. It then checks the answer and escalates to a stronger plan only when confidence is low. Every answer comes with a routing receipt explaining each choice.

Built for OxHack 2026.

## Quick start (about 15 minutes)

1. **Install**
   ```bash
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -e ".[demo]"
   cp .env.example .env
   ```
2. **Local model**: install [Ollama](https://ollama.com/download), then
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
3. **Keys in `.env`**: `GEMINI_API_KEY_1` from [AI Studio](https://aistudio.google.com/apikey) and `TAVILY_API_KEY` from [Tavily](https://app.tavily.com). Both have free tiers.
4. **Tiger Cloud (optional)**: create a free service, put its connection string in `DATABASE_URL`, then
   ```bash
   psql "$DATABASE_URL" -f db/schema.sql
   python -m scripts.ingest --reset
   ```
   Without `DATABASE_URL`, documents are indexed in memory and telemetry goes to `results/telemetry.jsonl`.
5. **Check everything is reachable**
   ```bash
   python -m scripts.check        # status + three test questions with their receipts
   python -m scripts.calibrate    # suggests RETRIEVAL_THRESHOLD for your embedder
   ```

## Run it

```bash
streamlit run tempo/dashboard/app.py     # the demo: side by side, receipts, results
uvicorn tempo.api.main:app --reload      # the API: POST /ask, POST /run_plan, GET /health
```

## Evaluate (the overnight batch)

```bash
python -m tempo.eval.batch --limit 3     # smoke test first
python -m tempo.eval.batch --resume      # 40 prompts x 4 systems, safe to stop and resume
python -m tempo.eval.report              # results/summary.md, summary.json, tradeoff.png
```

The test set (`data/testset.jsonl`) has 40 labelled prompts in five categories:

| Category | Prompts | What it tests |
| --- | --- | --- |
| simple_fact | 10 | Cheap local model is enough |
| private_docs | 10 | Retrieval from the 12 documents in `data/docs` |
| recent_news | 8 | Web search for events after the models' knowledge cutoff |
| multi_step | 6 | Maths word problems that need the larger model |
| personal_data | 6 | Personal data stays on the local model |

Tempo is compared with three fixed baselines: always the largest cloud model, always the cheapest model, and always retrieve.

Costs are list-price equivalents for the tokens used, even when you run on the free tier.

## How it works

```
Request (+ deadline, budget, privacy)
  -> Signals: difficulty, freshness vs model cutoff, personal data, private-index match
  -> Rules planner: picks a plan and writes a reason for every choice
  -> Plan runner: retrieval / web search, then the model, within the deadline
  -> Verifier: two samples must agree (and be grounded in retrieved text)
  -> Escalation: local -> light -> large -> add search, while time remains
  -> Telemetry: every attempt logged to Tiger Data
```

Override rules, in order:

1. **Personal data** (emails, phone, card, NI, NHS numbers, postcodes, date of birth) goes to the local model only. It is never web-searched and never escalated to the cloud.
2. **Private-document match** turns on retrieval and prefers the local model, so documents stay on the machine.
3. **Fresh questions** use web search. A question counts as fresh when it mentions a date after the model's cutoff or uses words like "latest" or "today".
4. **Maths, code or hard reasoning** goes to the larger cloud model.
5. **Everything else** goes to the cheapest model fast enough for the deadline. Latency is tracked live.

The deadline and budget can then drop web search or downgrade the model.

Not in this build: the agent loop (multi-step requests get a single call), the learned planner, prompt splitting and shadow checks.

## Layout

| Path | What |
| --- | --- |
| `tempo/core` | Schemas, engine (plan, run, verify, escalate, log), factory |
| `tempo/executors` | Ollama, Gemini (key rotation and backoff), Tavily search, fake client |
| `tempo/router` | Signal extractors, rules planner, escalation ladder |
| `tempo/verifier` | Sample agreement, grounding, confidence |
| `tempo/telemetry` | Embeddings, vector store (Tiger or memory), telemetry sinks |
| `tempo/eval` | Scoring, batch runner, report |
| `tempo/dashboard` | Streamlit demo |
| `scripts` | `check`, `calibrate`, `ingest` |
| `db/schema.sql` | Tiger Cloud schema: pgvectorscale DiskANN index and telemetry hypertable |

## CI/CD

Every push runs `.github/workflows/ci.yml`:

1. **Lint**: ruff check and ruff format.
2. **Unit tests**: about 50 tests with fake models, so CI needs no API keys. Then a uvicorn smoke test.
3. **Integration tests**: the Tiger layer against `timescaledb-ha`, with pgvector, pgvectorscale and TimescaleDB, after applying `db/schema.sql`.
4. **Docker**: build the image, run it and check `/health`.
5. **Publish**: on `main` only, the image goes to `ghcr.io/zeyad2591/tempo---ai`.

Before pushing, run:

```bash
ruff check . && ruff format --check . && pytest -m "not integration"
```

Unit tests must never call real APIs. Use `tempo.executors.fake.FakeClient` and `tests/conftest.py:make_engine`.
