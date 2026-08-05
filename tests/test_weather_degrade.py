"""날씨 조회 실패·결측을 다루는 방식 테스트.

날씨는 장소 선택에 가중치를 주는 재료이지 일정의 전제가 아니다. 같은
그래프의 다른 노드가 실패를 저하로 흡수하는 것과 같은 계약을 지키는지,
그리고 반영하지 못한 사실이 사용자에게 전달되는지를 본다.

다루는 범위:
  - hub 가 오류를 주거나 닿지 않을 때 잡을 실패시키지 않는가
  - 그 사실이 경고로 남는가
  - 예보가 비어 있는 날이 조용히 "맑음"으로 흡수되지 않는가
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import httpx

from app import agent_dependencies as deps
from app.nodes.agent_nodes import fetch_weather
from app.schemas.agent_schemas import AgentRequest, DateRange, Mobility


def _request(days: int = 3) -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 8, 5),
            date_end=date(2026, 8, 5 + days - 1),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
        mobility=Mobility.walk,
    )


class _Hub:
    """fetch_weather 만 흉내 내는 hub 대역."""

    def __init__(self, result) -> None:
        self._result = result

    async def fetch_weather(self, *_a, **_k):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _run(monkeypatch, result) -> dict:
    monkeypatch.setattr(
        deps, "get_hub_client", lambda: _Hub(result)
    )
    state = {"request": _request()}
    return asyncio.run(fetch_weather(state))


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://hub/v1/weather")
    response = httpx.Response(code, text="boom", request=request)
    return httpx.HTTPStatusError("err", request=request,
                                 response=response)


def test_http_error_does_not_fail_the_job(monkeypatch):
    """hub 가 오류를 줘도 추천을 중단하지 않는다.

    부가 정보 하나 때문에 사용자가 아무것도 못 받는 상황을 막는다.
    """
    out = _run(monkeypatch, _status_error(500))
    assert not out.get("error")
    assert out.get("weather") is None


def test_unreachable_hub_leaves_a_warning(monkeypatch):
    """닿지 않을 때도 마찬가지이며, 반영 못 했다는 사실을 남긴다."""
    out = _run(monkeypatch, httpx.ConnectError("down"))
    assert not out.get("error")
    warnings = out.get("warnings") or []
    assert any("날씨" in w for w in warnings)


def test_region_not_found_is_degraded_too(monkeypatch):
    """지역을 못 찾은 404 도 일정 생성을 막지 않는다."""
    out = _run(monkeypatch, _status_error(404))
    assert not out.get("error")
    assert (out.get("warnings") or []) != []


def test_missing_days_are_reported_not_absorbed(monkeypatch):
    """예보가 빈 날은 경고로 알린다.

    비어 있는 날은 아래 계산에서 강수확률 0 으로 처리되는데, 알리지
    않으면 "비 안 옴"과 구분되지 않는다.
    """
    out = _run(monkeypatch, {
        "daily": [{"date": "2026-08-05", "precipitation_prob": 20}],
        "missing_dates": ["2026-08-06", "2026-08-07"],
    })
    warnings = out.get("warnings") or []
    assert any("2026-08-06" in w and "2026-08-07" in w for w in warnings)


def test_missing_days_outside_the_trip_are_ignored(monkeypatch):
    """여행 기간 밖의 결측은 알릴 이유가 없다."""
    out = _run(monkeypatch, {
        "daily": [{"date": "2026-08-05", "precipitation_prob": 20}],
        "missing_dates": ["2026-09-01"],
    })
    assert (out.get("warnings") or []) == []


def test_complete_forecast_adds_no_warning(monkeypatch):
    """전부 채워졌으면 경고를 만들지 않는다."""
    out = _run(monkeypatch, {
        "daily": [
            {"date": "2026-08-05", "precipitation_prob": 20},
            {"date": "2026-08-06", "precipitation_prob": 10},
            {"date": "2026-08-07", "precipitation_prob": 0},
        ],
        "missing_dates": [],
    })
    assert (out.get("warnings") or []) == []
    assert out["weather"]["daily"][0]["precipitation_prob"] == 20


def test_malformed_missing_dates_do_not_break_the_node(monkeypatch):
    """결측 목록이 예상 밖 형태여도 노드가 죽지 않는다."""
    out = _run(monkeypatch, {
        "daily": [],
        "missing_dates": ["not-a-date", None],
    })
    assert not out.get("error")
    assert (out.get("warnings") or []) == []
