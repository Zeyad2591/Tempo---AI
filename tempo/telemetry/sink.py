"""Telemetry (T-16): every attempt is logged, to Tiger Data or to a JSONL file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tempo.core.schemas import TelemetryRow


class TelemetrySink:
    async def log(self, row: TelemetryRow) -> None:
        raise NotImplementedError


class NullTelemetry(TelemetrySink):
    def __init__(self) -> None:
        self.rows: list[TelemetryRow] = []

    async def log(self, row: TelemetryRow) -> None:
        self.rows.append(row)


class JsonlTelemetry(TelemetrySink):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def log(self, row: TelemetryRow) -> None:
        line = row.model_dump_json() + "\n"
        await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)


class TigerTelemetry(TelemetrySink):
    """Writes to the `telemetry` hypertable from db/schema.sql."""

    def __init__(self, database_url: str):
        self.url = database_url

    async def log(self, row: TelemetryRow) -> None:
        await asyncio.to_thread(self._insert, row)

    def _insert(self, row: TelemetryRow) -> None:
        import psycopg

        p = row.plan
        with psycopg.connect(self.url, autocommit=True) as c:
            c.execute(
                "INSERT INTO telemetry (ts, request_id, prompt_id, system, attempt, model, "
                "search, retrieve, location, latency_ms, cost, confidence, escalated, correct, "
                "plan) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    row.ts,
                    row.request_id,
                    row.prompt_id,
                    row.system,
                    row.attempt,
                    p.model,
                    p.search,
                    p.retrieve,
                    p.location,
                    row.latency_ms,
                    row.cost,
                    row.confidence,
                    row.escalated,
                    row.correct,
                    json.dumps(p.model_dump()),
                ),
            )
