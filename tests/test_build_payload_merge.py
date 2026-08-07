"""build_payload 병합 테스트 — 강화 산출물이 장소에 실리는 경로.

추천 이유(reasons)와 블로그 요약(summaries)은 서로 독립적으로 같은 장소에
병합된다. 블로그 요약은 별도 파이프라인으로 빠져 이 그래프에서는 채워지지
않지만, 그 형태로 저장된 옛 결과를 다시 조립할 때 값이 살아 있어야 하므로
병합 경로 자체는 남아 있다 — 이 파일이 그 경로를 지킨다.
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import pytest
from pydantic import ValidationError

from app.nodes.agent_nodes import build_payload
from app.schemas.agent_schemas import AgentRequest, DateRange, Place


def _request() -> AgentRequest:
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
    )


def _places(n: int = 2) -> list[Place]:
    return [
        Place(
            place_id=i,
            day=1,
            name=f"장소{i}",
            address="주소",
            lat=37.5,
            lng=127.0,
            recommended_visit_time="오전",
            content_id=f"c{i}",
        )
        for i in range(n)
    ]


def _state(**over) -> dict:
    st = {"job_id": "j", "request": _request(), "places": _places()}
    st.update(over)
    return st


def test_merges_bullets() -> None:
    """요약이 Place.bullets 로 병합된다."""
    out = asyncio.run(build_payload(_state(summaries={0: ["요약0-1", "요약0-2"]})))
    assert out["places"][0].bullets == ["요약0-1", "요약0-2"]
    assert out["places"][1].bullets is None


def test_merges_reason_and_bullets_together() -> None:
    """이유와 요약은 독립적으로 같은 장소에 함께 병합된다."""
    out = asyncio.run(build_payload(_state(
        reasons={0: "이유0"},
        summaries={0: ["요약0-1", "요약0-2"], 1: ["요약1-1", "요약1-2"]},
    )))
    assert out["places"][0].reason == "이유0"
    assert out["places"][0].bullets == ["요약0-1", "요약0-2"]
    assert out["places"][1].reason is None
    assert out["places"][1].bullets == ["요약1-1", "요약1-2"]


def test_skips_merge_on_error() -> None:
    """실패한 잡은 병합 없이 통과한다."""
    out = asyncio.run(build_payload(
        _state(error="failed", summaries={0: ["a", "b"]})
    ))
    assert out["places"][0].bullets is None


def test_passes_through_without_enhancements() -> None:
    """산출물이 없으면 장소를 그대로 통과시킨다."""
    state = _state()
    out = asyncio.run(build_payload(state))
    assert all(p.bullets is None and p.reason is None for p in out["places"])


def test_place_bullets_field_caps_at_two() -> None:
    """Place.bullets 자체도 두 건을 넘길 수 없다."""
    with pytest.raises(ValidationError):
        Place(
            place_id=0, day=1, name="장소", address="주소", lat=37.5, lng=127.0,
            recommended_visit_time="오전", bullets=["1", "2", "3"],
        )
