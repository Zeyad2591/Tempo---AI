"""Results (T-35): Tempo versus the baselines, as a table, JSON and a trade-off chart.

python -m tempo.eval.report   # reads results/run.jsonl, writes results/summary.{json,md,png}
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["tempo", "always_largest", "always_cheapest", "always_retrieve"]


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def summarise(rows: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    systems = {}
    for name, rs in by_sys.items():
        lat = [r["latency_ms"] for r in rs]
        cats: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            cats[r["category"]].append(r["correct"])
        routed = [r["route_ok"] for r in rs if r.get("route_ok") is not None]
        hard = [r for r in rs if r["category"] == "multi_step"]
        systems[name] = {
            "n": len(rs),
            "accuracy": round(100 * sum(r["correct"] for r in rs) / len(rs), 1),
            "median_ms": round(statistics.median(lat)),
            "p95_ms": round(pct(lat, 95)),
            "cost_per_1000": round(1000 * sum(r["cost"] for r in rs) / len(rs), 4),
            "escalation_rate": round(100 * sum(r["escalated"] for r in rs) / len(rs), 1),
            "route_accuracy": round(100 * sum(routed) / len(routed), 1) if routed else None,
            "hard_on_cheap_path": round(
                100 * sum(r["plan"]["model"] != "large" for r in hard) / len(hard), 1
            )
            if hard and name == "tempo"
            else None,
            "by_category": {c: round(100 * sum(v) / len(v), 1) for c, v in sorted(cats.items())},
        }
    out = {"systems": systems}
    t, b = systems.get("tempo"), systems.get("always_largest")
    if t and b:
        out["tempo_vs_largest"] = {
            "median_latency_change_pct": round(100 * (t["median_ms"] / b["median_ms"] - 1), 1),
            "p95_latency_change_pct": round(100 * (t["p95_ms"] / b["p95_ms"] - 1), 1),
            "cost_change_pct": round(100 * (t["cost_per_1000"] / b["cost_per_1000"] - 1), 1)
            if b["cost_per_1000"]
            else None,
            "accuracy_gap_points": round(t["accuracy"] - b["accuracy"], 1),
        }
    return out


def markdown(summary: dict) -> str:
    s = summary["systems"]
    names = [n for n in ORDER if n in s] + [n for n in s if n not in ORDER]
    lines = [
        "| System | Accuracy | Median latency | p95 latency | Cost per 1,000 | Escalated |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        x = s[n]
        lines.append(
            f"| {n} | {x['accuracy']}% | {x['median_ms']} ms | {x['p95_ms']} ms | "
            f"${x['cost_per_1000']:.4f} | {x['escalation_rate']}% |"
        )
    v = summary.get("tempo_vs_largest")
    if v:
        lines += [
            "",
            f"Tempo vs always-largest: median latency {v['median_latency_change_pct']:+}%, "
            f"cost {v['cost_change_pct']:+}%, accuracy {v['accuracy_gap_points']:+} points.",
        ]
    t = s.get("tempo")
    if t and t.get("route_accuracy") is not None:
        lines.append(f"Tempo picked the expected route {t['route_accuracy']}% of the time.")
    return "\n".join(lines) + "\n"


def chart(summary: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    s = summary["systems"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for name, x in s.items():
        size = 80 + 4000 * x["cost_per_1000"] / max(
            0.0001, max(v["cost_per_1000"] for v in s.values())
        )
        ax.scatter(x["median_ms"], x["accuracy"], s=size, alpha=0.6 if name != "tempo" else 0.95)
        ax.annotate(
            name, (x["median_ms"], x["accuracy"]), xytext=(8, 6), textcoords="offset points"
        )
    ax.set_xlabel("Median latency (ms) — lower is better")
    ax.set_ylabel("Accuracy (%) — higher is better")
    ax.set_title("Latency, accuracy and cost (bubble size = cost per request)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/run.jsonl")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.run).read_text().splitlines() if x.strip()]
    summary = summarise(rows)
    out = Path(a.out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = markdown(summary)
    (out / "summary.md").write_text(md)
    print(md)
    if chart(summary, out / "tradeoff.png"):
        print(f"Chart: {out / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
