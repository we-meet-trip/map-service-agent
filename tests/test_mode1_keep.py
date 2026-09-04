"""재탐색에서 남길 장소(stage=mode1 + places) 고정·병합 테스트."""
from __future__ import annotations

import asyncio
from datetime import date, time

import pytest
from pydantic import ValidationError

from app.nodes.agent_nodes import (
    _merge_pinned,
    _pinned_places,
    _remaining_targets,
    parse_input,
    recommend_places,
)
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    Mobility,
    Place,
    SelectedPlace,
)


def _request(**overrides) -> AgentRequest:
    base = dict(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
        mobility=Mobility.walk,
    )
    base.update(overrides)
    return AgentRequest(**base)


def _selected(name: str, cid: str | None, day: int = 1) -> SelectedPlace:
    return SelectedPlace(
        name=name,
        address="주소",
        lat=37.5,
        lng=127.0,
        day=day,
        content_id=cid,
    )


def _place(place_id: int, name: str, cid: str | None, day: int = 1) -> Place:
    return Place(
        place_id=place_id,
        day=day,
        name=name,
        address="주소",
        lat=37.5,
        lng=127.0,
        recommended_visit_time="10:00",
        content_id=cid,
        grounded=True,
    )


def _state(req: AgentRequest, targets: list[int] | None = None) -> dict:
    state: dict = {"job_id": "j", "request": req}
    if targets is not None:
        state["plan"] = {
            "days": [
                {"day": i + 1, "target_stops": t}
                for i, t in enumerate(targets)
            ]
        }
    return state


def test_places_lower_bound_allows_single_keep() -> None:
    """남길 곳이 한 군데뿐인 재탐색도 정상 요청이다."""
    req = _request(stage="mode1", places=[_selected("A", "c1")])
    assert len(req.places) == 1


def test_route_still_requires_two_places() -> None:
    """스키마 하한을 낮춘 대신 동선 경로의 2개 하한은 입력 검증이 지킨다."""
    req = _request(stage="route", places=[_selected("A", "c1")])
    out = asyncio.run(parse_input(_state(req)))
    assert out["error"] == "stage=route requires at least 2 places"

    ok = _request(
        stage="route", places=[_selected("A", "c1"), _selected("B", "c2")]
    )
    assert asyncio.run(parse_input(_state(ok))).get("error") is None


def test_pinned_only_for_mode1_with_places() -> None:
    """init/route 요청은 남길 장소로 보지 않는다."""
    keep = [_selected("A", "c1")]
    assert _pinned_places(_state(_request(stage="mode1", places=keep)))
    assert not _pinned_places(_state(_request(stage="init")))
    assert not _pinned_places(_state(_request(stage="mode1")))


def test_remaining_targets_reduces_plan_by_pinned_count() -> None:
    """남길 장소 수만큼 그날 뽑을 수를 줄인다 — 안 줄이면 일정이 부푼다."""
    state = _state(_request(stage="mode1"), targets=[4, 4])
    pinned = [
        _place(0, "A", "c1", day=1),
        _place(1, "B", "c2", day=1),
        _place(2, "C", "c3", day=2),
    ]

    assert _remaining_targets(state, pinned) is True
    assert [d["target_stops"] for d in state["plan"]["days"]] == [2, 3]


def test_remaining_targets_false_when_pins_fill_every_day() -> None:
    """남길 장소가 그날 몫을 채우면 더 뽑을 자리가 없다."""
    state = _state(_request(stage="mode1"), targets=[2])
    pinned = [_place(0, "A", "c1"), _place(1, "B", "c2")]

    assert _remaining_targets(state, pinned) is False
    assert state["plan"]["days"][0]["target_stops"] == 0


def test_recommend_places_skips_llm_when_pins_fill_plan() -> None:
    """더 뽑을 자리가 없으면 모델을 부르지 않고 남길 장소만 세운다."""
    req = _request(
        stage="mode1",
        places=[_selected("A", "c1"), _selected("B", "c2")],
    )
    state = _state(req, targets=[2])
    # 후보가 있어도 선정 경로로 새지 않아야 한다.
    state["candidates"] = [{"content_id": "c9", "name": "다른곳"}]

    out = asyncio.run(recommend_places(state))

    assert [p.name for p in out["places"]] == ["A", "B"]
    assert [p.place_id for p in out["places"]] == [0, 1]
    assert out.get("error") is None


def test_merge_pinned_dedupes_sorts_and_renumbers() -> None:
    """같은 곳이 두 번 들어가지 않고, 일차 순서대로 번호가 다시 매겨진다."""
    pinned = [_place(0, "A", "c1", day=2)]
    fresh = [
        _place(0, "새B", "c2", day=1),
        _place(1, "A중복", "c1", day=2),
        _place(2, "새C", None, day=2),
    ]

    merged = _merge_pinned(pinned, fresh)

    assert [p.name for p in merged] == ["새B", "A", "새C"]
    assert [p.place_id for p in merged] == [0, 1, 2]
    assert [p.day for p in merged] == [1, 2, 2]


def test_contract_bounds_unchanged_for_upper_limit() -> None:
    """상한 10개는 그대로다."""
    with pytest.raises(ValidationError):
        _request(stage="mode1", places=[_selected("A", "c1")] * 11)
