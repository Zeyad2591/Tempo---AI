"""Overnight batch (T-33, T-34): every test prompt through Tempo and the fixed baselines.

    python -m tempo.eval.batch                         # all systems, all prompts
    python -m tempo.eval.batch --limit 5 --systems tempo,always_largest
    python -m tempo.eval.batch --resume                # skip rows already in the output

Results append to results/run.jsonl, one row per (prompt, system). Safe to stop and resume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from tempo.core.engine import Tempo
from tempo.core.schemas import Plan, Request, Result
from tempo.eval.scoring import Item, is_correct, load_testset, route_ok

BASELINES: dict[str, Plan] = {
    "always_largest": Plan(model="large", location="cloud", reason="baseline: always largest"),
    "always_cheapest": Plan(model="local", location="local", reason="baseline: always cheapest"),
    "always_retrieve": Plan(
        model="light", retrieve=True, location="cloud", reason="baseline: always retrieve"
    ),
}
ALL_SYSTEMS = ["tempo", *BASELINES]


async def run_one(engine: Tempo, item: Item, system: str, deadline_ms: int) -> Result:
    req = Request(text=item.prompt, deadline_ms=deadline_ms, prompt_id=item.id)
    if system == "tempo":
        return await engine.ask(req, system="tempo")
    plan = BASELINES[system]
    if plan.model not in engine.available:
        fallback = "light" if "light" in engine.available else "local"
        plan = plan.model_copy(update={"model": fallback, "reason": plan.reason + " (fallback)"})
    return await engine.run_fixed(req, plan, system=system)


def row(item: Item, system: str, r: Result) -> dict:
    final_plan = r.plan
    return {
        "ts": datetime.now(UTC).isoformat(),
        "id": item.id,
        "category": item.category,
        "system": system,
        "prompt": item.prompt,
        "answer": r.answer,
        "correct": is_correct(r.answer, item),
        "route_ok": route_ok(final_plan, item) if system == "tempo" else None,
        "first_route_ok": route_ok(r.attempts[0].plan, item)
        if system == "tempo" and r.attempts
        else None,
        "latency_ms": r.latency_ms,
        "cost": r.cost,
        "confidence": r.confidence,
        "escalated": r.escalated,
        "attempts": len(r.attempts),
        "plan": final_plan.model_dump(),
        "first_plan": r.attempts[0].plan.model_dump() if r.attempts else None,
        "receipt": r.receipt,
        "sources": r.sources,
    }


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def main(args: argparse.Namespace) -> None:
    from tempo.core.factory import build_engine

    engine = await build_engine()
    print("Status:", engine.status())
    items = load_testset(Path(args.testset))
    if args.categories:
        items = [i for i in items if i.category in args.categories.split(",")]
    items = items[: args.limit] if args.limit else items
    systems = args.systems.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if args.resume and out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["id"], d["system"]))
    elif out.exists() and not args.append:
        out.rename(out.with_suffix(f".{int(time.time())}.bak.jsonl"))

    total = len(items) * len(systems)
    n = 0
    for item in items:
        for system in systems:
            n += 1
            if (item.id, system) in done:
                continue
            for attempt in range(3):
                try:
                    r = await run_one(engine, item, system, args.deadline_ms)
                    break
                except Exception as e:  # keep the batch going
                    print(f"  retry {attempt + 1} {item.id}/{system}: {e}")
                    await asyncio.sleep(5 * (attempt + 1))
            else:
                continue
            rec = row(item, system, r)
            await asyncio.to_thread(append_line, out, json.dumps(rec))
            mark = "ok " if rec["correct"] else "BAD"
            print(
                f"[{n}/{total}] {mark} {item.id:4} {system:16} {rec['latency_ms']:6} ms "
                f"${rec['cost']:.6f} {Plan(**rec['plan']).label()}"
            )
            await asyncio.sleep(args.pause)
    print(f"Done. Rows in {out}. Next: python -m tempo.eval.report")


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/testset.jsonl")
    ap.add_argument("--out", default="results/run.jsonl")
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--categories", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline-ms", type=int, default=8000)
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between calls (rate limits)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--append", action="store_true")
    asyncio.run(main(ap.parse_args()))


if __name__ == "__main__":
    cli()
