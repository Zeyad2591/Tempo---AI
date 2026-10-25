"""Rules planner with override rules and routing receipts (T-23, T-24).

Order of precedence:
  1. Personal data -> local model, cloud is forbidden (search only with a redacted query).
  2. Private-document match -> retrieve; prefer local so documents stay on the machine.
  3. Fresh question (after the model's cutoff) -> web search.
  4. Maths, code or hard reasoning -> larger cloud model.
  5. Otherwise the cheapest model that is fast enough for the deadline.
  Deadline and budget then trim the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tempo.config import ModelSpec
from tempo.core.schemas import Plan, Request, Signals

# Rough latency estimates (ms) used before live telemetry exists.
DEFAULT_LATENCY = {"local": 2500, "light": 1200, "large": 3500, "search": 1500, "retrieve": 150}


@dataclass
class LatencyTracker:
    """Exponentially weighted moving average of observed latency per component."""

    alpha: float = 0.3
    ewma: dict[str, float] = field(default_factory=dict)

    def observe(self, key: str, ms: float) -> None:
        prev = self.ewma.get(key)
        self.ewma[key] = ms if prev is None else self.alpha * ms + (1 - self.alpha) * prev

    def get(self, key: str) -> float:
        return self.ewma.get(key, DEFAULT_LATENCY.get(key, 2000))

    def estimate(self, plan: Plan) -> float:
        t = self.get(plan.model)
        if plan.search:
            t += self.get("search")
        if plan.retrieve:
            t += self.get("retrieve")
        return t


def estimate_cost(spec: ModelSpec, plan: Plan, prompt_chars: int) -> float:
    tokens_in = prompt_chars / 4 + (1200 if plan.retrieve or plan.search else 0) + 120
    tokens_out = 350
    return (tokens_in * spec.price_in + tokens_out * spec.price_out) / 1e6


class RulesPlanner:
    def __init__(
        self,
        models: dict[str, ModelSpec],
        latency: LatencyTracker,
        retrieval_threshold: float = 0.55,
    ):
        self.models = models
        self.latency = latency
        self.retrieval_threshold = retrieval_threshold

    def plan(self, req: Request, s: Signals, available: set[str], can_search: bool) -> Plan:
        why: list[str] = []
        has_local = "local" in available
        cloud = [m for m in ("light", "large") if m in available]
        private = req.private or bool(s.pii)

        # 1. privacy
        if private:
            if has_local:
                model = "local"
                what = ", ".join(s.pii) if s.pii else "caller marked request private"
                why.append(f"local model: personal data detected ({what}), cloud not allowed")
            elif cloud:
                model = cloud[0]
                why.append(
                    f"{model} cloud model: personal data detected but no local model, "
                    "so it is redacted before the call"
                )
            else:
                model = "local"
                why.append("no model reachable: add OPENROUTER_API_KEY or a Gemini key to .env")
        # 4. hard -> large
        elif s.difficulty >= 0.5 and "large" in available:
            model = "large"
            kind = "maths/code" if s.math_or_code else "hard reasoning"
            why.append(f"large model: {kind} (difficulty {s.difficulty:.2f})")
        elif s.difficulty >= 0.3 and "light" in available:
            model = "light"
            why.append(f"light cloud model: moderate difficulty ({s.difficulty:.2f})")
        else:
            model = self._fast_cheap(req, available, why)

        # 2. private documents
        threshold = self.retrieval_threshold - (0.1 if s.domain_hint else 0.0)
        retrieve = s.domain_score >= threshold
        if retrieve:
            why.append(f"retrieve: private index match {s.domain_score:.2f} >= {threshold:.2f}")
            if model != "large" and has_local and not private:
                model = "local"
                why.append("local model: keeps company documents on this machine")
        elif s.domain_score > 0:
            why.append(f"no retrieval: best index match {s.domain_score:.2f} < {threshold:.2f}")

        # 3. freshness
        search = False
        if retrieve and s.freshness >= 0.5:
            why.append("no web search: question is about company documents")
        elif private and s.freshness >= 0.5:
            why.append("no web search: personal data never leaves this machine")
        elif s.freshness >= 0.5 and can_search:
            search = True
            why.append("web search: " + "; ".join(s.fresh_reasons[:2]))
            if model == "local" and "light" in available:
                model = "light"
                why.append("light cloud model: summarising search results")
        elif s.freshness >= 0.5:
            why.append("fresh question but no search API configured")

        plan = Plan(
            model=model,
            search=search,
            retrieve=retrieve,
            agent=False,
            location=self.models[model].location if model in self.models else "local",
        )

        # deadline
        est = self.latency.estimate(plan)
        if est > req.deadline_ms:
            if plan.search and s.freshness < 0.9:
                plan.search = False
                why.append(f"dropped search: estimate {est:.0f} ms > deadline {req.deadline_ms} ms")
            if plan.model == "large" and "light" in available and not private:
                plan.model = "light"
                why.append(f"downgraded to light model to meet {req.deadline_ms} ms deadline")

        # budget
        spec = self.models.get(plan.model)
        if spec and estimate_cost(spec, plan, len(req.text)) > req.budget:
            for cheaper in ("light", "local"):
                if cheaper in available and cheaper != plan.model:
                    if cheaper == "light" and private:
                        continue
                    plan.model = cheaper
                    why.append(f"downgraded to {cheaper}: estimated cost over budget")
                    break

        if s.multi_step:
            why.append("multi-step request: agent loop not enabled in this build, single call")
        plan.location = self.models[plan.model].location if plan.model in self.models else "local"
        plan.reason = "; ".join(why) or "default plan"
        return plan

    def _fast_cheap(self, req: Request, available: set[str], why: list[str]) -> str:
        if "local" in available and self.latency.get("local") <= 0.5 * req.deadline_ms:
            why.append(
                f"local model: simple request, local ~{self.latency.get('local'):.0f} ms fits "
                f"the {req.deadline_ms} ms deadline at zero cost"
            )
            return "local"
        if "light" in available:
            why.append("light cloud model: simple request, local model too slow or unavailable")
            return "light"
        if "local" in available:
            why.append("local model: only model available")
            return "local"
        if "large" in available:
            why.append("large model: only model available")
            return "large"
        why.append("no model reachable: add OPENROUTER_API_KEY or a Gemini key to .env")
        return "local"


def next_stronger(plan: Plan, available: set[str], private: bool, can_search: bool) -> Plan | None:
    """Escalation ladder (T-15): local -> light -> large; at the top, add search if possible.

    Requests with personal data never escalate: every step up would leave the machine.
    """
    if private:
        return None
    order = ["local", "light", "large"]
    idx = order.index(plan.model) if plan.model in order else 0
    for m in order[idx + 1 :]:
        if m in available and not private:
            return plan.model_copy(
                update={"model": m, "location": "cloud", "reason": f"escalated from {plan.model}"}
            )
    if not plan.search and can_search:
        return plan.model_copy(
            update={"search": True, "reason": f"escalated: added web search to {plan.model}"}
        )
    return None
