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
from app.agent_settings import get_settings
from app.nodes.agent_nodes import (
    _place_from_candidate,
    recommend_places,
    search_places,
)
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    InventedPlace,
    InventedPlaces,
    Mobility,
    PlaceSelection,
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


    async def estimate_dwell(self, places):
        # 체류시간은 hub 룰이 정한다. 여기서는 요청한 장소마다 같은 값을
        # 돌려주어 시각 계산이 결정적으로 돌아가게만 한다.
        return {
            "estimates": [
                {
                    "content_id": p["content_id"],
                    "stay_minutes": 30,
                    "source": "category",
                }
                for p in places
            ]
        }

class _FakeGemini:
    """generate_structured 가 미리 정한 결과를 돌려주는 대역."""

    def __init__(self, result):
        self._result = result

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None, usage_sink=None
    ):
        return self._result


def test_place_from_candidate_grounded():
    """후보 dict 가 grounded Place 로 변환된다."""
    p = _place_from_candidate(0, _candidate(1), "오전", 1)
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
    p = _place_from_candidate(0, kakao, "오전 10시", 1)
    assert p.category == "음식점 / 카페"
    assert p.name == "우스블랑 청담점"
    assert p.grounded is True


def test_place_from_candidate_normalizes_visit_time():
    """방문시간(LLM 출력)에 꺾쇠가 있어도 정규화되어 통과한다."""
    p = _place_from_candidate(0, _candidate(1), "9시 > 11시", 1)
    assert p.recommended_visit_time == "9시 / 11시"


def test_search_places_degraded_on_error():
    """hub 호출 실패 시 error 없이 grounded=False 로 저하한다."""
    deps.set_hub_client(_FakeHub(exc=httpx.ConnectError("down")))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out.get("error") is not None
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
            PlaceSelection(index=2, day=1, recommended_visit_time="오전"),
            PlaceSelection(index=0, day=1, recommended_visit_time="오후"),
            PlaceSelection(index=99, day=1, recommended_visit_time="범위밖"),
            PlaceSelection(index=2, day=1, recommended_visit_time="중복"),
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


def test_no_candidates_fails_without_llm_invention():
    spy = _SchemaSpyGemini(None)
    deps.set_gemini_client(spy)
    try:
        out = asyncio.run(recommend_places({"job_id": "j", "request": _request()}))
    finally:
        deps.reset_all()
    assert "insufficient verified places" in out["error"]
    assert not out.get("places")
    assert spy.schemas == []


def test_search_places_degraded_on_value_error():
    """200 비-JSON 등 ValueError 도 하드실패 없이 저하한다."""
    deps.set_hub_client(_FakeHub(exc=ValueError("bad json")))
    try:
        state = {"job_id": "j", "request": _request()}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()
    assert out.get("error") is not None
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
            PlaceSelection(index=0, day=1, recommended_visit_time="t0"),
            PlaceSelection(index=1, day=1, recommended_visit_time="t1"),
            PlaceSelection(index=2, day=1, recommended_visit_time="t2"),
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


class _SeqGemini:
    """호출 순서대로 다른 결과를 돌려주는 대역(선택 → 생성 폴백 검증용)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None, usage_sink=None
    ):
        self.calls += 1
        return self._results.pop(0)


def test_invalid_selection_fails_without_invention_even_with_budget():
    gemini = _SeqGemini([PlacesSelection(selections=[
        PlaceSelection(index=99, day=1, recommended_visit_time="오전")
    ])])
    deps.set_gemini_client(gemini)
    try:
        out = asyncio.run(recommend_places({"job_id": "j", "request": _request(),
                                           "candidates": [_candidate(0)], "grounded": True}))
    finally:
        deps.reset_all()
    assert "insufficient verified places" in out["error"]
    assert out["llm_calls_used"] == 1
    assert not out.get("places")


def test_select_empty_without_budget_still_fails():
    """예산이 남지 않으면 폴백하지 않고 기존대로 실패한다(예산 초과 방지)."""
    bad_selection = PlacesSelection(
        selections=[
            PlaceSelection(index=99, day=1, recommended_visit_time="오전")
        ]
    )
    gemini = _SeqGemini([bad_selection])
    deps.set_gemini_client(gemini)
    try:
        state = {
            "job_id": "j",
            "request": _request(),
            "candidates": [_candidate(0)],
            "grounded": True,
            # 선택 호출이 마지막 예산을 소비하도록 상한-1 을 미리 채운다
            # (상한이 바뀌어도 "선택 후 잔여 0" 조건이 유지된다).
            "llm_calls_used": (
                get_settings().GEMINI_MAX_CALLS_PER_REQUEST - 1
            ),
        }
        out = asyncio.run(recommend_places(state))
    finally:
        deps.reset_all()

    assert "insufficient verified places" in out["error"]
    assert gemini.calls == 1


class _RecordingHub:
    """테마별 조회를 기록하는 대역(팬아웃 검증용). 검색어마다 다른 후보 반환."""

    def __init__(self, per_keyword: dict):
        self._per_keyword = per_keyword
        self.keywords: list = []

    async def search_places(
        self, province, city, *, mobility=None, keyword=None, size=15
    ):
        self.keywords.append(keyword)
        return {"places": self._per_keyword.get(keyword, [])}


def test_search_places_fans_out_per_theme_and_dedupes():
    """테마마다 따로 조회하고 content_id 로 합친다(모든 테마 반영).

    예전에는 테마를 공백으로 이어붙여 검색어 하나로 보냈고, Kakao 가 그
    문자열을 통째로 매칭해 조합에 따라 0건이 나왔다.
    """
    shared = _candidate(0)          # 두 테마 결과에 공통으로 등장 → 1건으로 합침
    only_cafe = _candidate(1)
    hub = _RecordingHub({
        "맛집": [shared],
        "카페": [shared, only_cafe],
    })
    deps.set_hub_client(hub)
    try:
        req = _request().model_copy(update={"theme": ["food", "cafe"]})
        state = {"job_id": "j", "request": req}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()

    # 테마 코드가 한국어 검색어로 매핑되어 각각 조회됐다.
    assert hub.keywords == ["맛집", "카페"]
    # content_id 중복은 제거된다.
    ids = [c["content_id"] for c in out["candidates"]]
    assert ids == [shared["content_id"], only_cafe["content_id"]]
    assert out["grounded"] is True


def test_search_places_without_theme_queries_region_once():
    """테마가 없으면 검색어 없이 1회만 조회한다(hub 가 행정구역명으로 검색)."""
    hub = _RecordingHub({None: [_candidate(0)]})
    deps.set_hub_client(hub)
    try:
        req = _request().model_copy(update={"theme": None})
        state = {"job_id": "j", "request": req}
        out = asyncio.run(search_places(state))
    finally:
        deps.reset_all()

    assert hub.keywords == [None]
    assert len(out["candidates"]) == 1


def test_theme_keywords_maps_known_and_keeps_unknown():
    """알려진 테마는 한국어로 매핑하고, 모르는 값은 원문을 그대로 쓴다."""
    from app.nodes.agent_nodes import _theme_keywords

    assert _theme_keywords(["food", "night"]) == ["맛집", "야경"]
    assert _theme_keywords(["산책"]) == ["산책"]
    assert _theme_keywords([]) == [None]
    assert _theme_keywords(None) == [None]
    # 같은 검색어로 수렴하는 중복은 한 번만 조회한다.
    assert _theme_keywords(["food", "food"]) == ["맛집"]


class _SchemaSpyGemini:
    """어떤 응답 스키마로 호출됐는지 기록하는 대역."""

    def __init__(self, result) -> None:
        self._result = result
        self.schemas: list[type] = []

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None, usage_sink=None
    ):
        self.schemas.append(schema)
        return self._result



def test_slim_schema_has_no_downstream_only_fields():
    """모델이 채울 수 없는 필드는 스키마에 두지 않는다."""
    fields = set(InventedPlace.model_fields)
    assert fields == {
        "day", "name", "address", "lat", "lng", "recommended_visit_time",
    }
    # 실측 출처·후속 단계가 채우는 값이 새어 들어오지 않았는지 본다.
    for leaked in ("place_id", "content_id", "source", "grounded",
                   "reason", "bullets", "stay_minutes", "visit_start"):
        assert leaked not in fields
