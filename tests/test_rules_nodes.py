"""rules_filter / score_and_rank 활성 노드 단위 테스트.

외부 호출 없이 HubClient 대역을 주입해 반경 필터·실내 보너스 랭킹의
계약(저하 규칙·순서 보존·PoP 추출)을 검증한다. 두 노드는 그래프에서
단순 add_edge 로 연결돼 있어 어떤 경우에도 `state["error"]` 를 세우면
안 된다(저하 = pass-through).
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import httpx
import pytest

from app import agent_dependencies as deps
from app.nodes.agent_nodes import rules_filter, score_and_rank
from app.schemas.agent_schemas import AgentRequest, DateRange, Mobility


def _request(**over) -> AgentRequest:
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
    base.update(over)
    return AgentRequest(**base)


def _candidate(i: int, **over) -> dict:
    c = {
        "content_id": f"c{i}",
        "source": "kakao",
        "name": f"장소{i}",
        "address": "주소",
        "lat": 37.5 + 0.01 * i,
        "lng": 127.0 + 0.01 * i,
    }
    c.update(over)
    return c


class _FilterHub:
    """filter_mobility_radius 만 흉내 내는 대역."""

    def __init__(self, *, filtered=None, exc=None) -> None:
        self._filtered = filtered
        self._exc = exc
        self.called_with = None

    async def filter_mobility_radius(self, origin, mobility, candidates):
        self.called_with = {
            "origin": origin,
            "mobility": mobility,
            "candidates": candidates,
        }
        if self._exc is not None:
            raise self._exc
        return {"filtered": self._filtered, "radius_m": 3000, "dropped": 0}


class _ScoreHub:
    """score_indoor_bonus 만 흉내 내는 대역."""

    def __init__(self, *, scored=None, exc=None) -> None:
        self._scored = scored if scored is not None else []
        self._exc = exc
        self.called_with = None

    async def score_indoor_bonus(self, pois, day_pop_max):
        self.called_with = {"pois": pois, "day_pop_max": day_pop_max}
        if self._exc is not None:
            raise self._exc
        return {"scored": self._scored}


def _run(node, state) -> dict:
    try:
        return asyncio.run(node(state))
    finally:
        deps.reset_all()


# ── rules_filter ──────────────────────────────────────────────

def test_rules_filter_keeps_subset() -> None:
    """hub 가 부분집합을 돌려주면 그 후보만 남는다."""
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    hub = _FilterHub(filtered=[cands[0], cands[2]])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(rules_filter, state)
    assert out.get("error") is None
    assert [c["content_id"] for c in out["candidates"]] == ["c0", "c2"]
    assert out.get("grounded") is True
    # origin 은 좌표 평균, mobility 는 요청값 그대로 전달.
    assert hub.called_with["mobility"] == "walk"
    assert set(hub.called_with["origin"]) == {"lat", "lng"}
    assert hub.called_with["origin"]["lat"] == pytest.approx(37.51)


def test_rules_filter_degrades_on_connect_error() -> None:
    """hub 연결 오류 시 후보 불변 + error 미설정(저하 통과)."""
    cands = [_candidate(0), _candidate(1)]
    deps.set_hub_client(_FilterHub(exc=httpx.ConnectError("down")))
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(rules_filter, state)
    assert out.get("error") is None
    assert out["candidates"] == cands
    assert out.get("grounded") is True


def test_rules_filter_empty_degrades_to_invent() -> None:
    """전량 탈락 시 candidates=[] + grounded=False 로 폴백한다."""
    cands = [_candidate(0), _candidate(1)]
    deps.set_hub_client(_FilterHub(filtered=[]))
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(rules_filter, state)
    assert out.get("error") is None
    assert out["candidates"] == []
    assert out["grounded"] is False


def test_rules_filter_none_mobility_passthrough() -> None:
    """이동수단 미지정이면 hub 호출 없이 통과한다."""
    cands = [_candidate(0)]
    hub = _FilterHub(filtered=[])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(mobility=None),
             "candidates": cands, "grounded": True}
    out = _run(rules_filter, state)
    assert out["candidates"] == cands
    assert hub.called_with is None  # hub 미호출


def test_rules_filter_transit_forwarded_and_unchanged() -> None:
    """transit 은 hub 로 전달되며 반경 무제한(전량 통과) 시 후보 불변.

    None 과 달리 transit 은 agent 가 임의로 건너뛰지 않고 hub 에 위임한다
    (hub 가 radius_m=null 로 무제한 처리 — frozen 계약). 결과적으로
    후보는 그대로 유지된다.
    """
    cands = [_candidate(0), _candidate(1)]
    hub = _FilterHub(filtered=cands)
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(mobility=Mobility.transit),
             "candidates": cands, "grounded": True}
    out = _run(rules_filter, state)
    assert hub.called_with["mobility"] == "transit"
    assert [c["content_id"] for c in out["candidates"]] == ["c0", "c1"]


def test_rules_filter_not_grounded_passthrough() -> None:
    """grounded=False 면 hub 호출 없이 통과(폴백 경로엔 실측 후보 없음)."""
    hub = _FilterHub(filtered=[])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(),
             "candidates": [_candidate(0)], "grounded": False}
    _run(rules_filter, state)
    assert hub.called_with is None


# ── score_and_rank ────────────────────────────────────────────

def test_score_reorders_by_score_desc() -> None:
    """점수 내림차순으로 후보가 안정 재정렬된다."""
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    hub = _ScoreHub(scored=[
        {"content_id": "c0", "score": 0.5},
        {"content_id": "c1", "score": 0.9},
        {"content_id": "c2", "score": 0.7},
    ])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(score_and_rank, state)
    assert out.get("error") is None
    assert [c["content_id"] for c in out["candidates"]] == ["c1", "c2", "c0"]
    assert out["scores"] == {"c0": 0.5, "c1": 0.9, "c2": 0.7}


def test_score_extracts_day_pop_max_and_indoor_flag() -> None:
    """weather.daily[].precipitation_prob 최댓값과 실내 판정을 hub 에 넘긴다."""
    cands = [_candidate(0, category_group_code="CE7"),
             _candidate(1, category_group_code="XX9")]
    hub = _ScoreHub(scored=[])
    deps.set_hub_client(hub)
    weather = {"daily": [
        {"date": "2026-07-06", "precipitation_prob": 20},
        {"date": "2026-07-07", "precipitation_prob": 80},
        {"date": "2026-07-08", "precipitation_prob": None},
    ]}
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True, "weather": weather}
    _run(score_and_rank, state)
    assert hub.called_with["day_pop_max"] == 80
    pois = hub.called_with["pois"]
    assert pois[0]["indoor_flag"] is True    # CE7 = 실내 그룹
    assert pois[1]["indoor_flag"] is False   # XX9 = 비실내
    assert pois[0]["base_score"] == pytest.approx(1.0)
    assert pois[1]["base_score"] == pytest.approx(0.99)


def test_score_degrades_on_error() -> None:
    """hub 오류 시 순서 불변 + error/scores 미설정(저하 통과)."""
    cands = [_candidate(0), _candidate(1)]
    deps.set_hub_client(_ScoreHub(exc=httpx.ConnectError("down")))
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(score_and_rank, state)
    assert out.get("error") is None
    assert [c["content_id"] for c in out["candidates"]] == ["c0", "c1"]
    assert "scores" not in out


def test_score_empty_scored_keeps_order() -> None:
    """빈 채점 응답이면 재정렬/scores 저장 없이 통과한다."""
    cands = [_candidate(0), _candidate(1)]
    deps.set_hub_client(_ScoreHub(scored=[]))
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(score_and_rank, state)
    assert [c["content_id"] for c in out["candidates"]] == ["c0", "c1"]
    assert "scores" not in out


def test_score_partial_keeps_relative_order() -> None:
    """일부만 점수가 있으면 미채점 후보는 상대 순서를 보존한다."""
    cands = [_candidate(0), _candidate(1), _candidate(2)]
    # c1 만 최상위 점수, c0/c2 는 응답에 없음(미채점).
    hub = _ScoreHub(scored=[{"content_id": "c1", "score": 5.0}])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(),
             "candidates": cands, "grounded": True}
    out = _run(score_and_rank, state)
    assert [c["content_id"] for c in out["candidates"]] == ["c1", "c0", "c2"]


def test_score_none_mobility_still_scores() -> None:
    """score_and_rank 는 이동수단과 무관하게(후보만 있으면) 채점한다."""
    cands = [_candidate(0), _candidate(1)]
    hub = _ScoreHub(scored=[{"content_id": "c1", "score": 9.0}])
    deps.set_hub_client(hub)
    state = {"job_id": "j", "request": _request(mobility=None),
             "candidates": cands, "grounded": True}
    out = _run(score_and_rank, state)
    assert [c["content_id"] for c in out["candidates"]] == ["c1", "c0"]
