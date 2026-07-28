"""B2 계약 확장(schedule_id/stage/exclude) + exclude 필터 테스트."""
from __future__ import annotations

import asyncio
from datetime import date, time

import pytest
from pydantic import ValidationError

from app import agent_dependencies as deps
from app.nodes.agent_nodes import search_places
from app.schemas.agent_schemas import AgentRequest, DateRange, Mobility


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


def _candidate(cid: str | None) -> dict:
    return {
        "content_id": cid,
        "source": "kakao",
        "name": f"장소{cid}",
        "address": "주소",
        "lat": 37.5,
        "lng": 127.0,
    }


class _FakeHub:
    def __init__(self, places: list) -> None:
        self._places = places

    async def search_places(
        self, province, city, *, mobility=None, keyword=None, size=15
    ):
        return {"places": self._places, "count": len(self._places)}


def _run_search(req: AgentRequest, places: list) -> dict:
    deps.set_hub_client(_FakeHub(places))
    try:
        return asyncio.run(search_places({"job_id": "j", "request": req}))
    finally:
        deps.reset_all()


def test_backward_compatible_defaults() -> None:
    """기존 body(신규 필드 없음)가 그대로 유효 — stage=init."""
    req = _request()
    assert req.stage == "init"
    assert req.schedule_id is None
    assert req.exclude is None


def test_exclude_filters_candidates() -> None:
    """exclude 의 content_id 후보가 제거되고 나머지는 유지된다."""
    req = _request(stage="mode1", exclude=["c1", "c3"])
    out = _run_search(
        req, [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    )
    assert out["grounded"] is True
    assert [c["content_id"] for c in out["candidates"]] == ["c2"]


def test_exclude_keeps_candidates_without_content_id() -> None:
    """content_id 없는 후보는 대조 불가라 통과된다."""
    req = _request(stage="mode1", exclude=["c1"])
    out = _run_search(req, [_candidate("c1"), _candidate(None)])
    assert [c["content_id"] for c in out["candidates"]] == [None]


def test_exclude_all_degrades_to_invent_path() -> None:
    """전량 제외 시 기존 저하 규칙(grounded=False)로 폴백한다."""
    req = _request(stage="mode1", exclude=["c1", "c2"])
    out = _run_search(req, [_candidate("c1"), _candidate("c2")])
    assert out["grounded"] is False
    assert out["candidates"] == []
    assert out.get("error") is None


def test_contract_bounds() -> None:
    """stage 오타/exclude 개수·항목 길이 상한이 스키마에서 거부된다."""
    with pytest.raises(ValidationError):
        _request(stage="mode2")
    with pytest.raises(ValidationError):
        _request(exclude=["x"] * 51)
    with pytest.raises(ValidationError):
        _request(exclude=["y" * 65])
    with pytest.raises(ValidationError):
        _request(schedule_id="s" * 65)
