"""Fake planner and executor that return the agreed shapes.

Build against these until the real router (P2) and executors (P1) land,
then swap the imports in tempo/api/main.py.
"""

from __future__ import annotations

import time

from tempo.core.schemas import Plan, Request, Result


def fake_plan(req: Request) -> Plan:
    if req.private:
        return Plan(model="local-small", location="local", reason="fake: private data -> local")
    return Plan(model="gemini-flash-lite", location="cloud", reason="fake: default light tier")


async def fake_execute(req: Request, plan: Plan) -> Result:
    start = time.perf_counter()
    answer = f"[fake answer to: {req.text[:60]}]"
    latency_ms = int((time.perf_counter() - start) * 1000)
    return Result(answer=answer, confidence=0.5, latency_ms=latency_ms, cost=0.0, plan=plan)
