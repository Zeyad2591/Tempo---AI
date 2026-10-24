# Tempo

A time-aware AI runtime that picks the fastest answer path that is still right, at the lowest cost. For each request it decides which model, whether to search the web, whether to retrieve from private documents, whether to run an agent, and whether to stay local or go to the cloud. Then it checks the answer and escalates only when needed.

Built for OxHack 2026.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
uvicorn tempo.api.main:app --reload
```

Then open http://localhost:8000/docs.

Or with Docker:

```bash
docker build -t tempo . && docker run -p 8000:8000 --env-file .env tempo
```

## Layout

| Folder | What lives there | Owner |
| --- | --- | --- |
| `tempo/core` | Shared schemas (`Request`, `Plan`, `Result`, `TelemetryRow`) and fakes | All |
| `tempo/api` | FastAPI app: `/health`, `/ask`, `/run_plan` | P1 |
| `tempo/executors` | Local, cloud, web search, retrieval, agent | P1 |
| `tempo/verifier` | Confidence checks and escalation | P1 |
| `tempo/router` | Signals, override rules, classifier, planners | P2 |
| `tempo/telemetry` | Tiger Data vector store and telemetry | P1, P3 |
| `tempo/eval` | Test set, scoring, baselines, overnight batch | P3 |
| `tempo/dashboard` | Side-by-side demo and routing receipts | P4 |

## CI/CD

Every push and pull request runs `.github/workflows/ci.yml` on GitHub Actions:

1. **Lint and format**: `ruff check` and `ruff format --check`.
2. **Unit tests**: `pytest -m "not integration"` with coverage, then boots the API with uvicorn and calls `/health` and `/ask`.
3. **Integration tests**: `pytest -m integration` against a `timescaledb-ha` Postgres service with pgvector, pgvectorscale and TimescaleDB, the same extensions as Tiger Cloud. Applies `db/schema.sql` first once it exists.
4. **Docker**: builds the image, runs it and checks `/health`.
5. **Publish** (pushes to `main` only, when everything above passes): pushes the image to `ghcr.io/zeyad2591/tempo---ai` tagged `latest` and the short commit SHA.

Check locally before pushing:

```bash
ruff check . && ruff format --check . && pytest -m "not integration"
```

Fix formatting automatically with `ruff format .` and most lint issues with `ruff check --fix .`.

### Rules for tests

- Unit tests must not call Gemini, Ollama, web search or Tiger Cloud. Mock them, so CI needs no secrets and never burns free-tier quota.
- Anything that needs a real database goes under `tests/integration/` and is marked `integration`.
