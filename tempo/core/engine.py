"""The Tempo engine: signals -> plan -> run -> verify -> escalate -> log (T-12, T-15)."""

from __future__ import annotations

import asyncio
import time
import uuid

from tempo.config import Settings
from tempo.core.schemas import Attempt, Plan, Request, Result, Signals, TelemetryRow
from tempo.executors.base import Completion, ModelClient, ProviderError
from tempo.executors.search import NoSearch, WebSearch
from tempo.router import signals as sig
from tempo.router.rules import LatencyTracker, RulesPlanner, next_stronger
from tempo.telemetry.sink import NullTelemetry, TelemetrySink
from tempo.telemetry.vectors import Passage, VectorStore
from tempo.verifier import verifier

SYSTEM = (
    "You are a precise assistant. Be concise and factual. Do not invent facts. "
    "If context is given, rely on it."
)


def build_prompt(req: Request, context: str, s: Signals) -> str:
    parts: list[str] = []
    if req.history:
        convo = "\n".join(f"{m.role}: {m.content}" for m in req.history[-6:])
        parts.append(f"Conversation so far:\n{convo}")
    if context:
        parts.append(
            "Answer using the context below. If the context does not contain the answer, say so."
            f"\n\nContext:\n{context}"
        )
    parts.append(f"Question: {req.text}")
    if s.math_or_code:
        parts.append("Work it out briefly, then end with one line: 'Final answer: <answer>'.")
    else:
        parts.append("Answer in one or two sentences. Put the answer first.")
    return "\n\n".join(parts)


class Tempo:
    def __init__(
        self,
        settings: Settings,
        clients: dict[str, ModelClient],
        search: WebSearch | None = None,
        store: VectorStore | None = None,
        telemetry: TelemetrySink | None = None,
    ):
        self.settings = settings
        self.clients = clients
        self.models = {n: c.spec for n, c in clients.items()}
        self.search = search or NoSearch()
        self.store = store
        self.telemetry = telemetry or NullTelemetry()
        self.latency = LatencyTracker()
        self.planner = RulesPlanner(self.models, self.latency, settings.retrieval_threshold)
        self.available: set[str] = set()
        self.can_search = False
        self.store_ready = False

    # ---------- setup ----------

    async def refresh(self) -> dict:
        names = list(self.clients)
        ups = await asyncio.gather(*(self.clients[n].available() for n in names))
        self.available = {n for n, ok in zip(names, ups, strict=True) if ok}
        self.can_search = await self.search.available()
        try:
            self.store_ready = self.store is not None and await self.store.count() > 0
        except Exception:
            self.store_ready = False
        return self.status()

    def status(self) -> dict:
        return {
            "models": {n: n in self.available for n in self.clients},
            "search": self.can_search,
            "retrieval": self.store_ready,
        }

    # ---------- public API ----------

    async def signals(self, req: Request) -> tuple[Signals, list[Passage]]:
        cutoffs = [self.models[m].cutoff for m in self.available] or [
            m.cutoff for m in self.models.values()
        ]
        passages: list[Passage] = []
        score = 0.0
        if self.store_ready:
            try:
                passages = await self.store.search(req.text, k=4)
                score = passages[0].score if passages else 0.0
            except Exception:
                passages = []
        return sig.extract(req.text, max(cutoffs), score), passages

    async def ask(self, req: Request, system: str = "tempo") -> Result:
        t0 = time.perf_counter()
        deadline = t0 + req.deadline_ms / 1000
        request_id = uuid.uuid4().hex[:12]
        s, passages = await self.signals(req)
        plan = self.planner.plan(req, s, self.available, self.can_search)
        private = req.private or bool(s.pii)
        receipt = [f"plan {plan.label()} ({plan.location}): {plan.reason}"]
        attempts: list[Attempt] = []
        sources: list[str] = []
        threshold = self.settings.confidence_threshold

        while True:
            nxt = next_stronger(plan, self.available, private, self.can_search)
            att, srcs = await self._attempt(
                req, plan, s, passages, deadline, samples=2 if nxt else 1, private=private
            )
            attempts.append(att)
            sources = srcs or sources
            receipt.append(
                f"attempt {len(attempts)}: {plan.label()} -> confidence {att.confidence:.2f} "
                f"in {att.latency_ms} ms ({att.note})"
            )
            await self._log(request_id, req, system, len(attempts), att, len(attempts) > 1)
            if att.confidence >= threshold:
                break
            if nxt is None:
                receipt.append("low confidence and no stronger plan available")
                break
            left_ms = (deadline - time.perf_counter()) * 1000
            if left_ms < self.latency.estimate(nxt):
                receipt.append(f"low confidence but only {left_ms:.0f} ms left: not escalating")
                break
            receipt.append(f"escalate: {att.confidence:.2f} < {threshold} -> {nxt.label()}")
            plan = nxt

        best = max(reversed(attempts), key=lambda a: a.confidence)
        return Result(
            answer=best.answer,
            confidence=best.confidence,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cost=round(sum(a.cost for a in attempts), 8),
            plan=best.plan,
            escalated=len(attempts) > 1,
            sources=sources,
            signals=s,
            attempts=attempts,
            receipt=receipt,
        )

    async def run_fixed(self, req: Request, plan: Plan, system: str = "fixed") -> Result:
        """Run one forced plan, no routing or escalation. Used by baselines and /run_plan."""
        t0 = time.perf_counter()
        s, passages = await self.signals(req)
        att, sources = await self._attempt(
            req, plan, s, passages, t0 + req.deadline_ms / 1000 + 60, samples=1, private=False
        )
        await self._log(uuid.uuid4().hex[:12], req, system, 1, att, False)
        return Result(
            answer=att.answer,
            confidence=att.confidence,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cost=att.cost,
            plan=plan,
            sources=sources,
            signals=s,
            attempts=[att],
            receipt=[f"forced plan {plan.label()} ({system})"],
        )

    # ---------- internals ----------

    async def _attempt(
        self,
        req: Request,
        plan: Plan,
        s: Signals,
        passages: list[Passage],
        deadline: float,
        samples: int,
        private: bool,
    ) -> tuple[Attempt, list[str]]:
        start = time.perf_counter()
        context_parts: list[str] = []
        sources: list[str] = []
        notes: list[str] = []

        if plan.retrieve:
            t = time.perf_counter()
            if not passages and self.store is not None:
                try:
                    passages = await self.store.search(req.text, k=4)
                except Exception as e:
                    notes.append(f"retrieval failed: {e}")
            self.latency.observe("retrieve", (time.perf_counter() - t) * 1000)
            for p in passages[:4]:
                context_parts.append(f"[{p.doc}] {p.text}")
                if p.doc not in sources:
                    sources.append(p.doc)

        if plan.search:
            t = time.perf_counter()
            query = sig.redact(req.text) if private else req.text
            try:
                hits = await asyncio.wait_for(self.search.search(query, k=4), timeout=8)
                for h in hits:
                    context_parts.append(f"[{h.title}] {h.content}")
                    sources.append(h.url)
            except Exception as e:
                notes.append(f"search failed: {type(e).__name__}")
            self.latency.observe("search", (time.perf_counter() - t) * 1000)

        context = "\n\n".join(context_parts)[:7000]
        prompt = build_prompt(req, context, s)
        client = self.clients.get(plan.model)
        if client is None:
            return self._failed(plan, start, f"model {plan.model} not configured"), sources

        temps = [0.2, 0.8][:samples]
        max_tokens = 700 if s.math_or_code else 300
        left = max(1.0, deadline - time.perf_counter())
        try:
            outs = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        client.generate(prompt, system=SYSTEM, temperature=t, max_tokens=max_tokens)
                        for t in temps
                    ),
                    return_exceptions=True,
                ),
                timeout=left,
            )
        except TimeoutError:
            return self._failed(plan, start, "deadline reached"), sources

        done: list[Completion] = [o for o in outs if isinstance(o, Completion)]
        if not done:
            err = next((o for o in outs if isinstance(o, BaseException)), None)
            msg = str(err)[:160] if isinstance(err, ProviderError) else type(err).__name__
            return self._failed(plan, start, f"model error: {msg}"), sources

        self.latency.observe(plan.model, done[0].latency_ms)
        conf, why = verifier.confidence([c.text for c in done], context)
        spec = self.models[plan.model]
        note = ", ".join([why, *notes])
        return (
            Attempt(
                plan=plan,
                answer=done[0].text.strip(),
                confidence=conf,
                latency_ms=int((time.perf_counter() - start) * 1000),
                cost=round(sum(c.cost(spec) for c in done), 8),
                note=note,
            ),
            sources,
        )

    def _failed(self, plan: Plan, start: float, note: str) -> Attempt:
        ms = int((time.perf_counter() - start) * 1000)
        return Attempt(plan=plan, answer="", confidence=0.0, latency_ms=ms, cost=0.0, note=note)

    async def _log(
        self, request_id: str, req: Request, system: str, n: int, att: Attempt, escalated: bool
    ) -> None:
        try:
            await self.telemetry.log(
                TelemetryRow(
                    request_id=request_id,
                    prompt_id=req.prompt_id,
                    system=system,
                    attempt=n,
                    plan=att.plan,
                    latency_ms=att.latency_ms,
                    cost=att.cost,
                    confidence=att.confidence,
                    escalated=escalated,
                )
            )
        except Exception:
            pass  # telemetry must never break a request
