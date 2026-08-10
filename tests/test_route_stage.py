"""stage="route" — 사용자가 고른 장소로 동선만 짜는 경로 테스트.

탐색·선정을 건너뛰고 받은 장소를 그대로 쓰는지, 그래서 LLM 호출이 초기
추천보다 한 번 적은지, 장소 목록이 없으면 실패로 끝나는지를 본다.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, time

import pytest
from pydantic import ValidationError

from app import agent_dependencies as deps
from app.graph.agent_graph import build_graph
from app.nodes.agent_nodes import load_given_places, parse_input
from app.schemas.agent_schemas import (
    AgentRequest,
    BulletsEnvelope,
    DateRange,
    Mobility,
    PlaceBullets,
    PlaceReason,
    ReasonEnvelope,
    SelectedPlace,
)


def _selected(n: int = 2) -> list[SelectedPlace]:
    return [
        SelectedPlace(
            name=f"내가고른곳{i}",
            address="주소",
            lat=37.5 + i * 0.01,
            lng=127.0 + i * 0.01,
            content_id=f"c{i}" if i == 0 else None,
        )
        for i in range(n)
    ]


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
        stage="route",
        places=_selected(),
    )
    base.update(over)
    return AgentRequest(**base)


class _FakeHub:
    """날씨·리뷰만 응답하는 대역. 장소 탐색이 불리면 즉시 드러나게 한다."""

    def __init__(self) -> None:
        self.search_calls = 0

    async def fetch_weather(self, province, city, date_start, date_end):
        return {
            "daily": [
                {"date": "2026-07-06", "precipitation_prob": 30}
            ],
            "missing_dates": [],
        }

    async def search_places(self, province, city, **kwargs):
        self.search_calls += 1
        return {"places": [], "count": 0}

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

    async def fetch_reviews(self, query, *, display=3):
        return {
            "reviews": [{"description": f"{query} 후기"}],
            "count": 1,
            "query": query,
        }


class _SeqGemini:
    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls = 0

    async def generate_structured(
        self, prompt, schema, *, system_instruction=None
    ):
        self.calls += 1
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _CapturePublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def publish(self, job_id, status, payload_json):
        self.payloads.append(json.loads(payload_json))
        return "1-0"


def _reason_envelope(n: int) -> ReasonEnvelope:
    return ReasonEnvelope(
        reasons=[PlaceReason(place_id=i, reason=f"이유{i}") for i in range(n)],
        clothing="가벼운 겉옷",
    )


def _bullets_envelope(n: int) -> BulletsEnvelope:
    return BulletsEnvelope(
        items=[
            PlaceBullets(place_id=i, bullets=[f"요약{i}-1", f"요약{i}-2"])
            for i in range(n)
        ]
    )


def _invoke(req, hub, gemini):
    pub = _CapturePublisher()
    deps.set_hub_client(hub)
    deps.set_gemini_client(gemini)
    deps.set_streams_publisher(pub)
    try:
        graph = build_graph()
        out = asyncio.run(graph.ainvoke({"job_id": "job-r", "request": req}))
    finally:
        deps.reset_all()
    return out, pub


# ── 요청 스키마 ────────────────────────────────────────────────

def test_places_require_two_to_ten() -> None:
    """장소가 1개면 이을 구간이 없고, 11개는 상한을 넘는다."""
    with pytest.raises(ValidationError):
        AgentRequest(
            date=DateRange(
                date_start=date(2026, 7, 6), date_end=date(2026, 7, 6),
                time_start=time(9, 0), time_end=time(18, 0),
            ),
            province="서울특별시", city="강남구",
            stage="route", places=_selected(1),
        )
    with pytest.raises(ValidationError):
        AgentRequest(
            date=DateRange(
                date_start=date(2026, 7, 6), date_end=date(2026, 7, 6),
                time_start=time(9, 0), time_end=time(18, 0),
            ),
            province="서울특별시", city="강남구",
            stage="route", places=_selected(11),
        )


def test_selected_place_rejects_out_of_range_and_tags() -> None:
    """국내 범위 밖 좌표와 태그 문자는 요청 단계에서 막는다."""
    with pytest.raises(ValidationError):
        SelectedPlace(name="해외", lat=10.0, lng=127.0)
    with pytest.raises(ValidationError):
        SelectedPlace(name="<script>", lat=37.5, lng=127.0)


# ── 노드 단위 ──────────────────────────────────────────────────

def test_parse_input_rejects_route_without_places() -> None:
    """stage=route 인데 장소가 없으면 실패로 표시한다."""
    req = AgentRequest(
        date=DateRange(
            date_start=date(2026, 7, 6), date_end=date(2026, 7, 6),
            time_start=time(9, 0), time_end=time(18, 0),
        ),
        province="서울특별시", city="강남구", stage="route",
    )
    out = asyncio.run(parse_input({"job_id": "j", "request": req}))
    assert out["error"] == "stage=route requires places"


def test_load_given_places_numbers_in_request_order() -> None:
    """받은 순서대로 place_id 0..N-1 을 매기고 grounded 로 둔다."""
    state = {"job_id": "j", "request": _request()}
    out = asyncio.run(load_given_places(state))
    places = out["places"]
    assert [p.place_id for p in places] == [0, 1]
    assert [p.name for p in places] == ["내가고른곳0", "내가고른곳1"]
    assert all(p.grounded for p in places)
    # 방문 시각은 BFF 가 활동 시간대를 나눠 채우므로 비워 둔다.
    assert all(p.recommended_visit_time == "" for p in places)
    assert out["grounded"] is True


def test_load_given_places_carries_category() -> None:
    """호출 측이 되돌려 준 분류를 그대로 싣는다.

    이 단계는 장소를 새로 찾지 않아 분류를 알아낼 길이 없다. 여기서 흘리면
    화면의 분류 칩이 사라진다.
    """
    given = [
        SelectedPlace(
            name="속초해변", lat=38.19, lng=128.60,
            category="여행 / 관광,명소 / 해수욕장,해변",
        ),
        SelectedPlace(name="영금정", lat=38.21, lng=128.60),
    ]
    state = {"job_id": "j", "request": _request(places=given)}
    places = asyncio.run(load_given_places(state))["places"]

    assert places[0].category == "여행 / 관광,명소 / 해수욕장,해변"
    assert places[1].category is None


def test_selected_place_tidies_category_instead_of_rejecting() -> None:
    """분류는 표시용이라 거절하지 않고 다듬는다.

    꺾쇠가 섞였다고 요청 전체를 막으면, 사용자가 본 적도 없는 글자 하나로
    동선을 못 만들게 된다. 뒤이어 Place 가 거부하는 문자만 바꾼다.
    """
    tagged = SelectedPlace(
        name="어딘가", lat=37.5, lng=127.0, category="음식점 > 카페",
    )
    assert tagged.category == "음식점 / 카페"

    long_one = SelectedPlace(
        name="어딘가", lat=37.5, lng=127.0, category="가" * 200,
    )
    assert len(long_one.category) == 80

    blank = SelectedPlace(name="어딘가", lat=37.5, lng=127.0, category="   ")
    assert blank.category is None


def test_load_given_places_noop_on_error() -> None:
    """앞 단계가 실패한 잡에서는 장소를 세우지 않는다."""
    state = {"job_id": "j", "request": _request(), "error": "boom"}
    out = asyncio.run(load_given_places(state))
    assert "places" not in out


# ── 그래프 통합 ────────────────────────────────────────────────

def test_route_stage_skips_search_and_selection() -> None:
    """장소 탐색을 부르지 않고 LLM 은 동선·이유 2회만 쓴다."""
    hub = _FakeHub()
    gemini = _SeqGemini([_reason_envelope(2)])

    out, pub = _invoke(_request(), hub, gemini)

    assert hub.search_calls == 0
    assert gemini.calls == 1
    assert out.get("error") is None
    payload = pub.payloads[0]
    assert payload["status"] == "done"
    assert [p["name"] for p in payload["places"]] == [
        "내가고른곳0", "내가고른곳1"
    ]
    assert payload["visit_order"] == [0, 1]
    assert len(payload["legs"]) == 1
    # 추천 이유는 초기 추천과 같은 방식으로 붙는다. 블로그 요약은 이 경로에
    # 실리지 않는다 — 장소를 눌렀을 때 별도 파이프라인이 만든다.
    assert payload["places"][0]["reason"] == "이유0"
    assert payload["places"][0]["bullets"] is None


def test_route_stage_stays_within_llm_budget() -> None:
    """route 경로의 LLM 소비는 예산 상한보다 작다(동선·이유 2회)."""
    from app.agent_settings import get_settings

    hub = _FakeHub()
    gemini = _SeqGemini([_reason_envelope(2)])
    out, _ = _invoke(_request(), hub, gemini)

    assert out["llm_calls_used"] == 1
    assert out["llm_calls_used"] <= get_settings().GEMINI_MAX_CALLS_PER_REQUEST


def test_init_stage_still_searches() -> None:
    """기본 stage 는 예전 그대로 탐색부터 한다(분기가 새지 않는다)."""
    hub = _FakeHub()
    gemini = _SeqGemini([RuntimeError("여기까지 오면 안 된다")])

    req = _request(stage="init", places=None)
    _invoke(req, hub, gemini)

    assert hub.search_calls == 1
