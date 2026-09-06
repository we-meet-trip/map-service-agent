"""출시 정책 회귀: HOLD, 실존 근거, 순서, 시간과 봉인 입력 경계."""
import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app import agent_dependencies as deps
from app.agent_settings import get_settings
from app.clients.agent_clients import StreamsPublisher, _record_usage
from app.nodes.agent_nodes import _training_signal, load_given_places, parse_input, recommend_route
from tests.test_route_stage import _request, _selected
from tests.test_timeline import _leg, _place, _request as timeline_request, _run, _state, _FakeHub
from eval.scorer import score_case, score_schema, score_timeline


def test_capture_disabled_blocks_generation_and_direct_publication(monkeypatch, caplog):
    monkeypatch.setattr(get_settings(), "TRAINING_CAPTURE_ENABLED", False)
    state = {"selection_path": "select", "candidates": [{"name": "private"}]}
    assert _training_signal(state) is None
    publisher = StreamsPublisher("redis://localhost", 0, "test")
    publisher._client = SimpleNamespace(xadd=AsyncMock(return_value="1-0"))
    asyncio.run(publisher.publish("job", "done", '{"status":"done"}', '{"private":true}'))
    fields = publisher._client.xadd.call_args.args[1]
    assert fields["status"] == "done" and "payload" in fields and "training" not in fields
    usage = []
    with caplog.at_level("INFO"):
        _record_usage(SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=3, candidates_token_count=2, total_token_count=5)), usage)
    assert usage == [{"prompt": 3, "output": 2, "total": 5}]
    assert "llm usage" in caplog.text


@pytest.mark.parametrize("optimize,expected", [(False, [0, 1, 2]), (True, [0, 2, 1])])
def test_manual_order_requires_explicit_optimization(optimize, expected):
    places = [_place(i).model_copy(update={"lat": 37.5, "lng": lng})
              for i, lng in enumerate([127.0, 127.02, 127.001])]
    state = {"job_id": "j", "request": _request(optimize=optimize), "places": places}
    out = asyncio.run(recommend_route(state))
    assert out["visit_order"] == expected
    assert [(leg.from_place_id, leg.to_place_id) for leg in out["legs"]] == list(zip(expected, expected[1:]))


def test_unverifiable_selected_place_fails():
    deps.set_hub_client(SimpleNamespace(search_places=AsyncMock(return_value={"places": []})))
    try:
        out = asyncio.run(load_given_places({"job_id": "j", "request": _request()}))
    finally:
        deps.reset_all()
    assert out["error"] and not out["places"] and not out["grounded"]


def test_selected_coordinates_are_replaced_by_verified_source():
    selected = _selected()
    canonical = [{**p.model_dump(), "content_id": p.content_id or "kakao:1", "source": "kakao"}
                 for p in selected]
    canonical[0]["lat"] = 37.501
    deps.set_hub_client(SimpleNamespace(search_places=AsyncMock(return_value={"places": canonical})))
    try:
        out = asyncio.run(load_given_places({"job_id": "j", "request": _request()}))
    finally:
        deps.reset_all()
    assert not out.get("error")
    assert out["places"][0].lat == 37.501


def test_selected_day_is_rejected_instead_of_clamped():
    selected = _selected()
    selected[0].day = 2
    out = asyncio.run(parse_input({"job_id": "j", "request": _request(places=selected)}))
    assert out["error"]


def test_impossible_time_budget_cannot_claim_trimmed_success():
    state = _state([_place(0), _place(1)], [_leg(0, 1, 200)], timeline_request(end_hour=10))
    out = _run(state, _FakeHub(60))
    assert out["error"] and out["timeline_status"] == "unverified"


def test_manual_time_overflow_keeps_selected_places():
    req = timeline_request(end_hour=10).model_copy(update={"stage": "route"})
    state = _state([_place(0), _place(1), _place(2)], [_leg(0, 1, 200), _leg(1, 2, 200)], req)
    out = _run(state, _FakeHub(60))
    assert out["error"] and out["visit_order"] == [0, 1, 2]
    assert len(out["places"]) == 3


@pytest.mark.parametrize("count", [0, 11])
def test_sealed_places_revalidate_outer_count(monkeypatch, count):
    from app.main import _resolve_places
    from app.crypto import location_seal
    monkeypatch.setattr(location_seal, "enabled", lambda: True)
    monkeypatch.setattr(location_seal, "open_seal", lambda _: {"places": [_selected()[0].model_dump()] * count})
    with pytest.raises(HTTPException):
        _resolve_places(_request(places=None, loc="test"))


@pytest.mark.parametrize("lat,lng", [(999, 127), (37.5, "bad"), (math.nan, 127)])
def test_evaluation_rejects_invalid_coordinates(lat, lng):
    payload = {"status": "done", "places": [{"place_id": 0, "day": 1, "name": "x", "lat": lat, "lng": lng}]}
    assert not score_schema(payload)["ok"]


def test_evaluation_uses_result_provenance_without_training():
    place = _place(0).model_dump()
    place.update(source="kakao", content_id="kakao:1", grounded=True)
    payload = {"status": "done", "places": [place], "visit_order": [0]}
    assert score_case({}, payload)["grounded"]["grounded"]
    place["grounded"] = False
    assert not score_case({}, payload)["grounded"]["grounded"]


def test_evaluation_checks_end_stay_overlap_and_missing_times():
    places = [_place(i).model_dump() for i in range(2)]
    places[0].update(visit_start="09:00", visit_end="09:50", stay_minutes=50)
    places[1].update(visit_start="09:59", visit_end="12:00", stay_minutes=121)
    payload = {"places": places, "visit_order": [0, 1], "legs": [{"from": 0, "to": 1, "estimated_duration_min": 20}]}
    result = score_timeline(payload, "09:00", "10:00")
    assert not result["ok"] and len(result["problems"]) >= 2
    places[1].update(visit_start=None, visit_end=None)
    assert not score_timeline(payload, "09:00", "10:00")["ok"]


def test_evaluation_cli_fails_without_baseline_and_does_not_replace_it(tmp_path, monkeypatch):
    import json
    from eval.run import main
    cases = tmp_path / "cases"
    results = tmp_path / "results"
    cases.mkdir()
    results.mkdir()
    (cases / "case.json").write_text(json.dumps({"id": "case", "expect": {}}))
    (results / "case.json").write_text(json.dumps({"status": "failed", "places": []}))
    monkeypatch.setattr("sys.argv", ["eval.run", "--cases", str(cases), "--results", str(results)])
    assert main() == 1
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"schema_ok_ratio": 1}')
    monkeypatch.setattr("sys.argv", ["eval.run", "--cases", str(cases), "--results", str(results),
                                     "--baseline", str(baseline), "--write-baseline"])
    assert main() == 1
    assert baseline.read_text() == '{"schema_ok_ratio": 1}'
