"""Pick RETRIEVAL_THRESHOLD for your embedder, and dry-run the router on the test set.

    python -m scripts.calibrate

Prints the best private-index match for every test prompt, suggests a threshold halfway
between document questions and everything else, then shows the plan for each prompt.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tempo.core.factory import build_engine
from tempo.core.schemas import Request
from tempo.eval.scoring import load_testset, route_ok


async def main() -> None:
    engine = await build_engine()
    print("Status:", engine.status())
    items = load_testset(Path("data/testset.jsonl"))
    doc, other = [], []
    for it in items:
        s, _ = await engine.signals(Request(text=it.prompt))
        (doc if it.category == "private_docs" else other).append(s.domain_score)
    if doc and other:
        lo, hi = max(other), min(doc)
        print(f"\nDoc questions: min match {hi:.3f}. Other questions: max match {lo:.3f}.")
        if hi > lo:
            print(f"Suggested RETRIEVAL_THRESHOLD={(hi + lo) / 2:.2f}  (put it in .env)")
        else:
            print(f"Ranges overlap; try RETRIEVAL_THRESHOLD={hi:.2f} and check misroutes below.")

    print(f"\nRouter dry run (current threshold {engine.settings.retrieval_threshold}):")
    bad = 0
    for it in items:
        req = Request(text=it.prompt)
        s, _ = await engine.signals(req)
        p = engine.planner.plan(req, s, engine.available, engine.can_search)
        ok = route_ok(p, it)
        bad += ok is False
        print(f"  {'ok ' if ok else 'BAD'} {it.id} {p.label():22} match={s.domain_score:.2f}")
    print(f"Misroutes: {bad}/{len(items)}")


if __name__ == "__main__":
    asyncio.run(main())
