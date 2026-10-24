"""Tempo dashboard (T-37 to T-42).

streamlit run tempo/dashboard/app.py
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import streamlit as st

from tempo.core.schemas import Plan, Request, Result
from tempo.eval.scoring import is_correct, load_testset

DEMO_IDS = ["f01", "f05", "d01", "d09", "n01", "n04", "m01", "m03", "p01", "p03"]
LARGEST = Plan(model="large", location="cloud", reason="baseline: always the biggest model")
RESULTS = Path("results")

st.set_page_config(page_title="Tempo", layout="wide")


# One event loop in a background thread, so async HTTP clients survive Streamlit reruns.
@st.cache_resource
def runtime():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    from tempo.core.factory import build_engine

    engine = asyncio.run_coroutine_threadsafe(build_engine(), loop).result(timeout=120)
    return loop, engine


def run(coro):
    loop, _ = runtime()
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=180)


loop, engine = runtime()


async def side_by_side(text: str, deadline_ms: int, budget: float) -> tuple[Result, Result]:
    req = Request(text=text, deadline_ms=deadline_ms, budget=budget)
    base_plan = (
        LARGEST if "large" in engine.available else LARGEST.model_copy(update={"model": "light"})
    )
    return await asyncio.gather(engine.ask(req), engine.run_fixed(req, base_plan, "always_largest"))


def badge(plan: Plan) -> str:
    bits = [f"**{plan.model}**", "local" if plan.location == "local" else "cloud"]
    if plan.search:
        bits.append("web search")
    if plan.retrieve:
        bits.append("retrieval")
    return " · ".join(bits)


def show_result(col, title: str, r: Result, correct: bool | None = None) -> None:
    with col:
        st.subheader(title)
        a, b, c = st.columns(3)
        a.metric("Latency", f"{r.latency_ms:,} ms")
        b.metric("Cost", f"${r.cost:.6f}")
        c.metric("Correct", "—" if correct is None else ("yes" if correct else "no"))
        st.markdown(badge(r.plan) + (" · escalated" if r.escalated else ""))
        st.info(r.answer or "(no answer)")
        with st.expander("Routing receipt"):
            for line in r.receipt:
                st.markdown(f"- {line}")
            if r.sources:
                st.markdown("Sources: " + ", ".join(r.sources))
            if r.signals:
                st.json(r.signals.model_dump(), expanded=False)


# ---------- sidebar ----------
with st.sidebar:
    st.title("Tempo")
    st.caption("Fastest answer path that is still right, at the lowest cost.")
    status = engine.status()
    for name, up in status["models"].items():
        st.markdown(f"{'🟢' if up else '🔴'} model `{name}`")
    st.markdown(f"{'🟢' if status['search'] else '🔴'} web search")
    st.markdown(f"{'🟢' if status['retrieval'] else '🔴'} private document index")
    if st.button("Re-check services"):
        run(engine.refresh())
        st.rerun()
    deadline = st.slider("Deadline (ms)", 1000, 20000, 8000, 500)
    budget = st.select_slider("Budget per request (USD)", [0.0001, 0.001, 0.01, 0.1], 0.01)

live, demo, results = st.tabs(["Ask", "Demo: 10 prompts", "Overnight results"])

# ---------- live ----------
with live:
    text = st.text_area("Question", "Who was elected Pope in May 2025?", height=80)
    if st.button("Run side by side", type="primary") and text.strip():
        with st.spinner("Running Tempo and the always-biggest baseline..."):
            t, b = run(side_by_side(text, deadline, budget))
        left, right = st.columns(2)
        show_result(left, "Tempo", t)
        show_result(right, "Always the biggest model", b)

# ---------- demo ----------
with demo:
    items = {i.id: i for i in load_testset(Path("data/testset.jsonl"))}
    picks = [items[i] for i in DEMO_IDS if i in items]
    st.caption("The same ten prompts through Tempo and through always-the-biggest-model.")
    if st.button("Run the demo", type="primary"):
        rows = []
        totals = {"tempo": [0, 0.0, 0], "largest": [0, 0.0, 0]}
        counters = st.empty()
        table = st.empty()
        for item in picks:
            t, b = run(side_by_side(item.prompt, deadline, budget))
            for key, r in (("tempo", t), ("largest", b)):
                totals[key][0] += r.latency_ms
                totals[key][1] += r.cost
                totals[key][2] += is_correct(r.answer, item)
            rows.append((item, t, b))
            with counters.container():
                c1, c2 = st.columns(2)
                for col, key, label in ((c1, "tempo", "Tempo"), (c2, "largest", "Always biggest")):
                    ms, cost, ok = totals[key]
                    col.markdown(f"### {label}")
                    m1, m2, m3 = col.columns(3)
                    m1.metric("Total time", f"{ms / 1000:.1f} s")
                    m2.metric("Total cost", f"${cost:.5f}")
                    m3.metric("Correct", f"{ok}/{len(rows)}")
            table.dataframe(
                [
                    {
                        "prompt": i.prompt[:70],
                        "tempo route": t.plan.label() + (" (escalated)" if t.escalated else ""),
                        "tempo ms": t.latency_ms,
                        "biggest ms": b.latency_ms,
                        "tempo ok": is_correct(t.answer, i),
                        "biggest ok": is_correct(b.answer, i),
                    }
                    for i, t, b in rows
                ],
                width="stretch",
            )
        st.session_state["demo_rows"] = rows
    for item, t, b in st.session_state.get("demo_rows", []):
        flag = " — escalated" if t.escalated else ""
        with st.expander(f"{item.prompt[:90]} → {t.plan.label()}{flag}"):
            left, right = st.columns(2)
            show_result(left, "Tempo", t, is_correct(t.answer, item))
            show_result(right, "Always the biggest model", b, is_correct(b.answer, item))

# ---------- results ----------
with results:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        st.warning(
            "No results yet. Run `python -m tempo.eval.batch` then `python -m tempo.eval.report`."
        )
    else:
        summary = json.loads(summary_path.read_text())
        v = summary.get("tempo_vs_largest")
        if v:
            a, b, c = st.columns(3)
            a.metric("Median latency vs biggest", f"{v['median_latency_change_pct']:+}%")
            b.metric(
                "Cost vs biggest",
                f"{v['cost_change_pct']:+}%" if v["cost_change_pct"] is not None else "—",
            )
            c.metric("Accuracy gap", f"{v['accuracy_gap_points']:+} pts")
        st.dataframe(
            [
                {
                    "system": n,
                    **{
                        k: x[k]
                        for k in (
                            "accuracy",
                            "median_ms",
                            "p95_ms",
                            "cost_per_1000",
                            "escalation_rate",
                        )
                    },
                }
                for n, x in summary["systems"].items()
            ],
            width="stretch",
        )
        st.markdown("**Accuracy by category (%)**")
        st.dataframe(
            [{"system": n, **x["by_category"]} for n, x in summary["systems"].items()],
            width="stretch",
        )
        t = summary["systems"].get("tempo", {})
        if t.get("route_accuracy") is not None:
            st.caption(
                f"Tempo picked the expected route {t['route_accuracy']}% of the time; "
                f"{t.get('hard_on_cheap_path')}% of hard prompts ended on a cheaper model."
            )
        if (RESULTS / "tradeoff.png").exists():
            st.image(str(RESULTS / "tradeoff.png"))
