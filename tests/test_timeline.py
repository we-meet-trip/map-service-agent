"""시간축 노드(build_timeline / fit_time_budget) 테스트.

일정에 "몇 시에 얼마나"를 세우는 구간이라 LLM 이 관여하지 않는다. 입력이
같으면 결과가 항상 같아야 하므로 시각을 값으로 못 박아 검증한다.

다루는 범위:
  - 방문 시각이 (체류 + 이동 + 여유) 누적으로 만들어지는가
  - 일차가 바뀌면 활동 시작 시각부터 다시 세는가
  - 하루 활동 시간을 넘치면 체류시간을 줄이고, 그래도 넘치면 덜어 내는가
  - 덜어 낸 뒤에도 "legs 길이 = 장소 수 - 1" 계약이 유지되는가
  - hub 가 죽으면 시각 없이 통과하고 표시만 남기는가
  - 영업시간을 확인할 수 없다는 사실이 항상 알려지는가
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import httpx

from app import agent_dependencies as deps
from app.nodes.agent_nodes import build_timeline, fit_time_budget
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    Leg,
    Mobility,
    Place,
)


def _request(*, end_hour: int = 18, days: int = 1) -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6 + days - 1),
            time_start=time(9, 0),
            time_end=time(end_hour, 0),
        ),
        province="서울특별시",
        city="강남구",
        mobility=Mobility.walk,
    )


def _place(pid: int, *, day: int = 1) -> Place:
    return Place(
        place_id=pid,
        day=day,
        name=f"장소{pid}",
        address="주소",
        lat=37.5 + pid * 0.01,
        lng=127.0 + pid * 0.01,
        recommended_visit_time="오전",
        content_id=f"c{pid}",
        category_group_code="AT4",
    )


def _leg(a: int, b: int, minutes: int) -> Leg:
    return Leg.model_validate(
        {
            "from": a,
            "to": b,
            "mode": "walk",
            "estimated_distance_km": 1.0,
            "estimated_duration_min": minutes,
        }
    )


class _FakeHub:
    """장소마다 정해진 체류시간을 돌려주는 hub 대역."""

    def __init__(self, minutes: int = 60, *, fail: bool = False) -> None:
        self._minutes = minutes
        self._fail = fail
        self.calls = 0

    async def estimate_dwell(self, places):
        self.calls += 1
        if self._fail:
            raise httpx.ConnectError("hub down")
        return {
            "estimates": [
                {
                    "content_id": p["content_id"],
                    "stay_minutes": self._minutes,
                    "source": "category",
                }
                for p in places
            ]
        }


def _run(state, hub):
    deps.set_hub_client(hub)
    try:
        out = asyncio.run(build_timeline(state))
        return asyncio.run(fit_time_budget(out))
    finally:
        deps.reset_all()


def _state(places, legs, req):
    return {
        "job_id": "j",
        "request": req,
        "places": places,
        "visit_order": [p.place_id for p in places],
        "legs": legs,
    }


def test_visit_times_accumulate_stay_travel_and_buffer() -> None:
    """방문 시각은 체류 + 이동 + 여유를 쌓아 만든다."""
    places = [_place(0), _place(1), _place(2)]
    legs = [_leg(0, 1, 20), _leg(1, 2, 15)]
    out = _run(_state(places, legs, _request()), _FakeHub(60))

    timeline = out["timeline"]
    # 09:00 시작 → 60분 체류 → 10:00 종료 → 이동 20 + 여유 10 → 10:30 시작
    assert timeline[0]["visit_start"] == "09:00"
    assert timeline[0]["visit_end"] == "10:00"
    assert timeline[1]["visit_start"] == "10:30"
    assert timeline[1]["visit_end"] == "11:30"
    # 이동 15 + 여유 10 → 11:55
    assert timeline[2]["visit_start"] == "11:55"
    assert timeline[2]["visit_end"] == "12:55"
    assert out["timeline_status"] == "ok"


def test_each_day_restarts_from_active_start_hour() -> None:
    """일차가 바뀌면 활동 시작 시각부터 다시 센다."""
    places = [_place(0, day=1), _place(1, day=1), _place(2, day=2)]
    legs = [_leg(0, 1, 20), _leg(1, 2, 30)]
    out = _run(_state(places, legs, _request(days=2)), _FakeHub(60))

    timeline = out["timeline"]
    assert timeline[1]["visit_end"] == "11:30"
    # 2일차 첫 장소는 전날 끝 시각을 잇지 않고 09:00 에서 다시 시작한다.
    assert timeline[2]["visit_start"] == "09:00"


def test_overflow_shrinks_stay_first() -> None:
    """활동 시간을 넘치면 먼저 체류시간을 하한까지 줄인다(장소는 유지)."""
    # 09:00~12:00(180분)에 장소 3곳 · 체류 60 · 이동 10 → 크게 초과
    places = [_place(0), _place(1), _place(2)]
    legs = [_leg(0, 1, 10), _leg(1, 2, 10)]
    out = _run(_state(places, legs, _request(end_hour=12)), _FakeHub(60))

    assert out["timeline_status"] == "trimmed"
    assert len(out["places"]) == 3
    assert all(
        slot["stay_minutes"] == 10 for slot in out["timeline"].values()
    )


def test_overflow_drops_last_places_when_shrink_is_not_enough() -> None:
    """하한까지 줄여도 넘치면 그날 뒤쪽부터 덜어 낸다."""
    # 09:00~10:00(60분)에 장소 4곳 · 구간마다 이동 30 → 줄여도 불가능
    places = [_place(i) for i in range(4)]
    legs = [_leg(0, 1, 30), _leg(1, 2, 30), _leg(2, 3, 30)]
    out = _run(_state(places, legs, _request(end_hour=10)), _FakeHub(60))

    assert out["timeline_status"] == "trimmed"
    assert len(out["places"]) < 4
    # 앞쪽 동선은 그대로 남는다.
    assert out["visit_order"][0] == 0


def test_dropping_keeps_leg_count_contract() -> None:
    """덜어 낸 뒤에도 legs 길이 = 장소 수 - 1 이 유지된다."""
    places = [_place(i) for i in range(4)]
    legs = [_leg(0, 1, 30), _leg(1, 2, 30), _leg(2, 3, 30)]
    out = _run(_state(places, legs, _request(end_hour=10)), _FakeHub(60))

    assert len(out["legs"]) == len(out["places"]) - 1
    order = out["visit_order"]
    assert len(order) == len(out["places"])
    # 각 구간이 방문 순서의 이웃을 정확히 잇는다.
    for i, leg in enumerate(out["legs"]):
        assert leg.from_place_id == order[i]
        assert leg.to_place_id == order[i + 1]


def test_hub_failure_degrades_without_error() -> None:
    """hub 가 죽어도 잡을 세우지 않고 표시만 남긴다."""
    places = [_place(0), _place(1)]
    legs = [_leg(0, 1, 20)]
    out = _run(_state(places, legs, _request()), _FakeHub(fail=True))

    assert out.get("error") is None
    assert out["timeline_status"] == "unverified"
    assert "timeline" not in out


def test_opening_hours_limit_is_always_disclosed() -> None:
    """영업시간을 확인할 출처가 없다는 사실은 성패와 무관하게 알린다."""
    places = [_place(0), _place(1)]
    legs = [_leg(0, 1, 20)]
    ok = _run(_state(places, legs, _request()), _FakeHub(60))
    degraded = _run(_state(places, legs, _request()), _FakeHub(fail=True))

    assert any("영업시간" in w for w in ok["warnings"])
    assert any("영업시간" in w for w in degraded["warnings"])


def test_transit_mobility_discloses_estimated_travel() -> None:
    """도로 경로를 못 받는 이동수단은 이동시간이 추정값임을 알린다."""
    req = _request()
    req = req.model_copy(update={"mobility": Mobility.transit})
    places = [_place(0), _place(1)]
    out = _run(_state(places, [_leg(0, 1, 20)], req), _FakeHub(60))

    assert any("추정" in w for w in out["warnings"])


def test_disabled_flag_keeps_payload_as_before() -> None:
    """스위치를 끄면 시각을 만들지 않아 도입 전과 같은 상태로 남는다."""
    from app.agent_settings import get_settings

    settings = get_settings()
    original = settings.TIMELINE_ENABLED
    settings.TIMELINE_ENABLED = False
    try:
        hub = _FakeHub(60)
        places = [_place(0), _place(1)]
        out = _run(_state(places, [_leg(0, 1, 20)], _request()), hub)
    finally:
        settings.TIMELINE_ENABLED = original

    assert hub.calls == 0
    assert "timeline" not in out
    assert "timeline_status" not in out


# ── 일차별 전략 ─────────────────────────────────────────────────

def test_plan_strategy_uses_each_days_precipitation() -> None:
    """실내 비중을 여행 전체가 아니라 그날 강수확률로 정한다."""
    from app.nodes.agent_nodes import plan_strategy

    state = {
        "job_id": "j",
        "request": _request(days=3),
        # hub 응답과 같은 형태로 날짜를 함께 싣는다 — 일차 매칭은 위치가
        # 아니라 날짜로 하므로, 날짜가 없으면 어느 날 값인지 알 수 없다.
        "weather": {
            "daily": [
                {"date": "2026-07-06", "precipitation_prob": 0},
                {"date": "2026-07-07", "precipitation_prob": 80},
                {"date": "2026-07-08", "precipitation_prob": 40},
            ]
        },
    }
    out = asyncio.run(plan_strategy(state))

    ratios = [d["indoor_ratio"] for d in out["plan"]["days"]]
    # 맑은 날은 실내 강제 없음, 비 오는 날만 실내를 늘린다.
    assert ratios[0] == 0.0
    assert ratios[1] > ratios[2] > ratios[0]


def test_plan_strategy_scales_stops_with_active_hours() -> None:
    """활동 시간이 짧으면 그날 목표 장소 수도 줄어든다."""
    from app.nodes.agent_nodes import plan_strategy

    short = asyncio.run(plan_strategy(
        {"job_id": "j", "request": _request(end_hour=11), "weather": {}}
    ))
    long = asyncio.run(plan_strategy(
        {"job_id": "j", "request": _request(end_hour=21), "weather": {}}
    ))

    assert short["plan"]["days"][0]["target_stops"] < \
        long["plan"]["days"][0]["target_stops"]


def test_plan_strategy_fills_missing_weather_days() -> None:
    """예보가 모자란 날도 전략은 여행 일수만큼 나온다."""
    from app.nodes.agent_nodes import plan_strategy

    out = asyncio.run(plan_strategy({
        "job_id": "j",
        "request": _request(days=4),
        "weather": {"daily": [{"precipitation_prob": 60}]},
    }))

    assert len(out["plan"]["days"]) == 4
    assert [d["day"] for d in out["plan"]["days"]] == [1, 2, 3, 4]


def test_plan_strategy_matches_weather_by_date_not_position() -> None:
    """예보가 비어 앞날이 빠져 오면, 그 자리를 뒤 날짜 값으로 메우지 않는다.

    hub 는 예보를 못 구한 날을 목록에서 빼고 보낸다. 배열 위치로 세면
    1일차가 나흘 뒤 강수확률을 받아, 맑은 날이 비 오는 날로 뒤바뀐다.
    """
    from app.nodes.agent_nodes import plan_strategy

    # 여행은 2026-07-06 부터 3일. 그런데 hub 가 앞 이틀을 못 주고
    # 사흘째와 그 뒤 날짜만 보냈다.
    out = asyncio.run(plan_strategy({
        "job_id": "j",
        "request": _request(days=3),
        "weather": {
            "daily": [
                {"date": "2026-07-08", "precipitation_prob": 90},
                {"date": "2026-07-09", "precipitation_prob": 80},
            ]
        },
    }))

    days = out["plan"]["days"]
    # 1·2일차는 예보가 없으므로 편향 없음. 3일차만 90 을 받는다.
    assert [d["precipitation_prob"] for d in days] == [0, 0, 90]
    assert days[0]["indoor_ratio"] == 0.0
    assert days[2]["indoor_ratio"] > 0.0
