"""FastAPI entry point. Run with: uvicorn tempo.api.main:app --reload"""

from __future__ import annotations

from fastapi import FastAPI

from tempo import __version__
from tempo.core.fakes import fake_execute, fake_plan
from tempo.core.schemas import Plan, Request, Result

app = FastAPI(title="Tempo", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=Result)
async def ask(req: Request) -> Result:
    """Router picks a plan, runner executes it. Uses fakes until T-12 and T-24 land."""
    plan = fake_plan(req)
    return await fake_execute(req, plan)


@app.post("/run_plan", response_model=Result)
async def run_plan(req: Request, plan: Plan) -> Result:
    """Force a specific plan. Used by baselines and the overnight batch."""
    return await fake_execute(req, plan)
