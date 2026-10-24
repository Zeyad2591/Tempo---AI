from tempo.core.schemas import Plan
from tempo.eval.report import markdown, summarise
from tempo.eval.scoring import Item, is_correct, load_testset, route_ok
from tests.conftest import ROOT


def test_testset_is_well_formed():
    items = load_testset(ROOT / "data" / "testset.jsonl")
    assert len(items) == 40
    assert len({i.id for i in items}) == 40
    cats = {i.category for i in items}
    assert cats == {"simple_fact", "private_docs", "recent_news", "multi_step", "personal_data"}
    assert all(i.answers for i in items)


def test_scoring_aliases_and_numbers():
    item = Item("x", "c", "q", ["Canberra"])
    assert is_correct("The capital is Canberra.", item)
    assert not is_correct("Sydney", item)
    num = Item("y", "c", "q", ["10.20"], numeric=10.2)
    assert is_correct("You save Final answer: £10.20", num)
    assert is_correct("You save £10.2 overall", num)
    assert not is_correct("You save £110.20", num)


def test_scoring_handles_accents():
    item = Item("z", "c", "q", ["María Corina Machado"])
    assert is_correct("Maria Corina Machado won it", item)


def test_route_ok():
    item = Item("p", "c", "q", ["a"], expect={"location": "local"})
    assert route_ok(Plan(model="local", location="local"), item) is True
    assert route_ok(Plan(model="light", location="cloud"), item) is False


def test_summary_compares_with_largest():
    def r(system, ms, cost, ok, cat="simple_fact", model="local"):
        return {
            "system": system,
            "latency_ms": ms,
            "cost": cost,
            "correct": ok,
            "category": cat,
            "escalated": False,
            "route_ok": True if system == "tempo" else None,
            "plan": {"model": model},
        }

    rows = [r("tempo", 500, 0.0, True), r("tempo", 900, 0.0001, True)]
    rows += [r("always_largest", 2000, 0.001, True, model="large")] * 2
    s = summarise(rows)
    v = s["tempo_vs_largest"]
    assert v["median_latency_change_pct"] < 0 and v["cost_change_pct"] < 0
    assert v["accuracy_gap_points"] == 0
    assert "tempo" in markdown(s)
