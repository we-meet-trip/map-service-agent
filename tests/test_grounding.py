"""grounding 경로(실측 후보 선택)와 저하 폴백의 단위 테스트.

외부 호출 없이 검증하기 위해 HubClient/GeminiClient 자리에 가짜
구현을 주입한다. 비동기 노드는 asyncio.run 으로 실행한다.
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import httpx
import pytest

from app import agent_dependencies as deps
from app.nodes.agent_nodes import (
    _place_from_candidate,
    recommend_places,
    search_places,
)
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    Mobility,
    Place,
    PlaceSelection,
    PlacesEnvelope,
    PlacesSelection,
)


def _request() -> AgentRequest:
    """테스트용 추천 요청 한 건."""
    return AgentRequest(
        date=DateRange(
            date_start=date(2026, 6, 1),
            date_end=date(2026, 6, 1),
            time_start=time(9, 0),
            time_end=time(18, 0),
        ),
        province="서울특별시",
        city="강남구",
        mobility=Mobility.walk,
        theme=["산책"],
    )


def _candidate(idx: int) -> dict:
    """테스트용 실측 후보 한 건."""
    return {
        "content_id": f"durunubi:{idx}",
        "source": "durunubi",
        "name": f"코스{idx}",
        "address": "서울 종로구",
        "lat": 37.59,
        "lng": 126.97,
        "category": "걷기길",
        "crs_dstnc_km": 8.0,
        "crs_level": 2,
        "brd_div": "DNWW",
        "gpx_url": "http://example/gpx",
        "route_idx": "r1",
    }


class _FakeHub:
    """search_places 만 흉내 내는 HubClient 대역."""

    def __init__(self, resp: dict | None = None, exc: Exception | None = None):
        self._resp = resp
        self._exc = exc

    async def search_places(
        self, province, city, *, mobility=None, keyword=None, size=15
    ):
        if self._exc is not None:
            raise self._exc
        return self._resp


class _FakeGemini:
    """generate_structured 가 미리 정한 결과를 돌려주는 대역."""

    def __init__(self, result):
        self._result = result

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None
    ):
        return self._result


def test_place_from_candidate_grounded():
    """후보 dict 가 grounded Place 로 변환된다."""
    p = _place_from_candidate(0, _candidate(1), "오전")
    assert p.grounded is True
    assert p.source == "durunubi"
    assert p.lat == pytest.approx(37.59)
    assert p.recommended_visit_time == "오전"
    assert p.gpx_url == "http://example/gpx"


def test_place_from_candidate_normalizes_kakao_category():
    """Kakao category 의 '>' 가 정규화되어 Place 생성이 성공한다.

    이전에는 category "음식점 > 카페" 의 '>' 가 Place 검증기에 걸려
    ValidationError → 후보 skip → grounded 추천 전멸이었다.
    """
    kakao = {
        "content_id": "kakao:1891511991",
        "source": "kakao",
        "name": "우스블랑 청담점",
        "address": "서울 강남구 청담동 44-14",
        "lat": 37.5182,
        "lng": 127.0454,
        "category": "음식점 > 카페",
        "category_group_code": "CE7",
    }
    p = _place_from_candidate(0, kakao, "오전 10시")
    assert p.category == "음식점 / 카페"
    assert p.name == "우스블랑 청담점"
    assert p.grounded is True


def test_place_from_candidate_normalizes_visit_time():
    """방문시간(LLM 출력)에 꺾쇠가 있어도 정규화되어 통과한다."""
    p = _place_from_candidate(0, _candidate(1), "9시 > 11시")
    assert p.recommended_visit_time == "9시 / 11시"


def test_search_places_degraded_on_error():
    """hub 호출 실패 시 error 없이 grounded=False 로 저하한다."""
    deps.set_hub_client(_FakeHub(exc=httpx.ConnectError("down")))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out.get("error") is None
    assert out["grounded"] is False
    assert out["candidates"] == []


def test_search_places_success():
    """후보가 있으면 grounded=True 로 채운다."""
    resp = {
        "places": [_candidate(1), _candidate(2)],
        "count": 2,
        "sources": {"durunubi": 2},
    }
    deps.set_hub_client(_FakeHub(resp=resp))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out["grounded"] is True
    assert len(out["candidates"]) == 2


def test_recommend_places_grounded_selection():
    """grounded 경로에서 유효 인덱스만 순서대로 선택된다."""
    candidates = [_candidate(i) for i in range(3)]
    selection = PlacesSelection(
        selections=[
            PlaceSelection(index=2, recommended_visit_time="오전"),
            PlaceSelection(index=0, recommended_visit_time="오후"),
            PlaceSelection(index=99, recommended_visit_time="범위밖"),
            PlaceSelection(index=2, recommended_visit_time="중복"),
        ]
    )
    deps.set_gemini_client(_FakeGemini(selection))
    try:
        state = {
            "job_id": "j",
            "request": _request(),
            "candidates": candidates,
            "grounded": True,
        }
        out = asyncio.run(recommend_places(state))
    finally:
        deps.reset_all()
    places = out["places"]
    assert [p.place_id for p in places] == [0, 1]
    assert places[0].name == "코스2"
    assert places[1].name == "코스0"
    assert all(p.grounded for p in places)


def test_recommend_places_fallback_invent():
    """후보가 없으면 LLM 생성 폴백 + grounded=False 표시."""
    invented = PlacesEnvelope(
        places=[
            Place(
                place_id=5,
                name="가짜",
                address="addr",
                lat=37.5,
                lng=127.0,
                recommended_visit_time="오전",
            )
        ]
    )
    deps.set_gemini_client(_FakeGemini(invented))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(recommend_places(state))
    finally:
        deps.reset_all()
    places = out["places"]
    assert len(places) == 1
    assert places[0].place_id == 0
    assert places[0].grounded is False


def test_search_places_degraded_on_value_error():
    """200 비-JSON 등 ValueError 도 하드실패 없이 저하한다."""
    deps.set_hub_client(_FakeHub(exc=ValueError("bad json")))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out.get("error") is None
    assert out["grounded"] is False
    assert out["candidates"] == []


def test_search_places_degraded_on_non_dict_body():
    """응답이 dict 가 아니면(예상 밖 형태) 저하한다(예외 미전파)."""
    deps.set_hub_client(_FakeHub(resp=["not", "a", "dict"]))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out["grounded"] is False
    assert out["candidates"] == []


def test_select_places_skips_bad_coord_candidate():
    """좌표 없는 후보가 선택되어도 건너뛰고 나머지로 진행한다."""
    candidates = [_candidate(0), {"name": "좌표없음", "source": "kakao"},
                  _candidate(2)]
    selection = PlacesSelection(
        selections=[
            PlaceSelection(index=0, recommended_visit_time="t0"),
            PlaceSelection(index=1, recommended_visit_time="t1"),
            PlaceSelection(index=2, recommended_visit_time="t2"),
        ]
    )
    deps.set_gemini_client(_FakeGemini(selection))
    try:
        state = {
            "job_id": "j",
            "request": _request(),
            "candidates": candidates,
            "grounded": True,
        }
        out = asyncio.run(recommend_places(state))
    finally:
        deps.reset_all()
    places = out["places"]
    assert [p.place_id for p in places] == [0, 1]
    assert places[0].name == "코스0"
    assert places[1].name == "코스2"
