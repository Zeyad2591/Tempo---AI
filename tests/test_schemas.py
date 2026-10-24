import pytest
from pydantic import ValidationError

from tempo.core.schemas import Plan, Request, Result, TelemetryRow


def test_request_defaults():
    req = Request(text="hi")
    assert req.deadline_ms > 0
    assert req.private is False
    assert req.history == []


@pytest.mark.parametrize("bad", [{"deadline_ms": 0}, {"budget": -1}, {"accuracy_target": 1.5}])
def test_request_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        Request(text="hi", **bad)


def test_result_confidence_bounds():
    plan = Plan(model="m", reason="r")
    with pytest.raises(ValidationError):
        Result(answer="a", confidence=1.2, latency_ms=1, cost=0, plan=plan)


def test_plan_location_is_local_or_cloud():
    with pytest.raises(ValidationError):
        Plan(model="m", location="moon")


def test_telemetry_row_round_trips():
    row = TelemetryRow(prompt_id="p1", plan=Plan(model="m"), latency_ms=12, cost=0.0001)
    assert TelemetryRow.model_validate_json(row.model_dump_json()) == row
