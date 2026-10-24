"""FastAPI entry point (T-17). Run with: uvicorn tempo.api.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import Request as HttpRequest

from tempo import __version__
from tempo.core.engine import Tempo
from tempo.core.schemas import Plan, Request, Result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set app.state.engine beforehand with fake clients.
    if getattr(app.state, "engine", None) is None:
        from tempo.core.factory import build_engine

        app.state.engine = await build_engine()
    yield


app = FastAPI(title="Tempo", version=__version__, lifespan=lifespan)


def engine(http: HttpRequest) -> Tempo:
    return http.app.state.engine


@app.get("/health")
def health(http: HttpRequest) -> dict:
    return {"status": "ok", "version": __version__, **engine(http).status()}


@app.post("/refresh")
async def refresh(http: HttpRequest) -> dict:
    """Re-check which models, search and retrieval are reachable."""
    return await engine(http).refresh()


@app.post("/ask", response_model=Result)
async def ask(req: Request, http: HttpRequest) -> Result:
    """Route, run, verify and escalate. The result carries its routing receipt."""
    return await engine(http).ask(req)


@app.post("/run_plan", response_model=Result)
async def run_plan(req: Request, plan: Plan, http: HttpRequest) -> Result:
    """Force a specific plan. Used by baselines and the overnight batch."""
    return await engine(http).run_fixed(req, plan, system="forced")
