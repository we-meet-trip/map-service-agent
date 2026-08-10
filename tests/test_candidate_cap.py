"""선정 프롬프트에 실을 후보 수를 상한까지 줄이는 동작.

검증 계약:
  - 상한 이하면 손대지 않는다.
  - 상한을 넘으면 실내/실외를 절반씩 남긴다(정렬이 없어도 한쪽이 통째로
    잘리지 않는다).
  - 한쪽이 모자라면 남은 자리를 다른 쪽으로 채워 상한을 정확히 맞춘다.
  - 뽑은 뒤 원래 순서(=순위)를 유지한다.
  - 상한은 여행 일수에 따라 늘어난다.
"""
from __future__ import annotations

import asyncio
from datetime import date, time

from app.nodes.agent_nodes import (
    _cap_candidates,
    _candidate_limit,
    recommend_places,
)
from app.schemas.agent_schemas import AgentRequest, DateRange

# 실내로 판정되는 카카오 카테고리 그룹 중 하나.
_INDOOR = "CE7"
_OUTDOOR = "AT4"


def _request(days: int = 1) -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6 + days - 1),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
    )


def _cands(spec: str) -> list[dict]:
    """"io" 같은 문자열로 실내(i)/실외(o) 후보 목록을 만든다."""
    return [
        {
            "content_id": f"c{i}",
            "name": f"장소{i}",
            "lat": 37.5,
            "lng": 127.0,
            "category_group_code": _INDOOR if ch == "i" else _OUTDOOR,
        }
        for i, ch in enumerate(spec)
    ]


def test_under_limit_is_untouched() -> None:
    """상한 이하면 원본 리스트를 그대로 돌려준다."""
    cands = _cands("ioio")
    assert _cap_candidates(cands, 10) is cands


def test_keeps_both_sides_when_over_limit() -> None:
    """상한을 넘으면 실내·실외를 절반씩 남긴다."""
    # 앞쪽에 실내만 20개, 뒤쪽에 실외만 20개 — 앞에서 그냥 자르면 실외가
    # 통째로 사라지는 배치다.
    kept = _cap_candidates(_cands("i" * 20 + "o" * 20), 10)
    codes = [c["category_group_code"] for c in kept]
    assert len(kept) == 10
    assert codes.count(_INDOOR) == 5
    assert codes.count(_OUTDOOR) == 5


def test_fills_from_other_side_when_short() -> None:
    """한쪽이 모자라면 다른 쪽으로 채워 상한을 정확히 맞춘다."""
    kept = _cap_candidates(_cands("i" * 20 + "o" * 2), 10)
    codes = [c["category_group_code"] for c in kept]
    assert len(kept) == 10
    assert codes.count(_OUTDOOR) == 2
    assert codes.count(_INDOOR) == 8


def test_original_order_is_preserved() -> None:
    """뽑은 뒤에도 원래 순서를 유지한다 — 순서가 곧 순위다."""
    kept = _cap_candidates(_cands("iooiiooi" * 5), 10)
    ids = [int(c["content_id"][1:]) for c in kept]
    assert ids == sorted(ids)


def test_limit_scales_with_trip_length() -> None:
    """상한은 하한과 일수 비례분 중 큰 쪽이다."""
    # 1일: 12 < 40 이므로 하한 40 이 이긴다.
    assert _candidate_limit({"request": _request(1)}) == 40
    # 5일: 60 > 40.
    assert _candidate_limit({"request": _request(5)}) == 60


class _SpyGemini:
    """선정 프롬프트가 몇 개의 후보를 받았는지 세는 대역."""

    def __init__(self, envelope) -> None:
        self.envelope = envelope
        self.prompts: list[str] = []

    async def generate_structured(
        self, prompt, response_schema, *, system_instruction=None
    ):
        self.prompts.append(prompt)
        return self.envelope


def test_recommend_places_caps_before_prompting() -> None:
    """노드를 통과하면 상한을 넘는 후보가 프롬프트에 실리지 않는다."""
    import json

    from app import agent_dependencies as deps
    from app.schemas.agent_schemas import PlaceSelection, PlacesSelection

    envelope = PlacesSelection(
        selections=[
            PlaceSelection(index=0, day=1, recommended_visit_time="오전"),
            PlaceSelection(index=1, day=1, recommended_visit_time="오후"),
        ]
    )
    spy = _SpyGemini(envelope)
    deps.set_gemini_client(spy)
    try:
        state = {
            "job_id": "job-1",
            "request": _request(1),
            "candidates": _cands("io" * 60),  # 120개
            "grounded": True,
        }
        asyncio.run(recommend_places(state))
    finally:
        deps.reset_all()

    body = spy.prompts[0]
    start = body.index("<candidates>") + len("<candidates>")
    end = body.index("</candidates>")
    assert len(json.loads(body[start:end])) == 40
    # state 에도 줄어든 목록이 반영된다.
    assert len(state["candidates"]) == 40
