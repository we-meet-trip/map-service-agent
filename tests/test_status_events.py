"""agent:jobs:status 진행 이벤트 발행 테스트.

검증 계약:
  - 정상 경로: stage 시퀀스가 그래프 노드 순서와 일치.
  - 에러 단축 경로: 건너뛴 노드의 stage 미발행.
  - 발행 실패/미주입: 잡은 정상 완료(모든 예외 삼킴).
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, time

from app import agent_dependencies as deps
from app.graph.agent_graph import build_graph
from app.schemas.agent_schemas import (
    AgentRequest,
    DateRange,
    Leg,
    Mobility,
    PlaceReason,
    PlaceSelection,
    PlacesSelection,
    ReasonEnvelope,
    RouteEnvelope,
)


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


class _FakeHub:
    async def fetch_weather(self, province, city, date_start, date_end):
        return {"daily": []}

    async def search_places(
        self, province, city, *, mobility=None, keyword=None, size=15
    ):
        return {
            "places": [
                {"content_id": "c0", "name": "장소0", "address": "주소",
                 "lat": 37.5, "lng": 127.0},
                {"content_id": "c1", "name": "장소1", "address": "주소",
                 "lat": 37.6, "lng": 127.1},
            ],
        }

    # 룰/리뷰 엔드포인트 무해 통과 대역 — 후보/순서/stage 시퀀스를 바꾸지
    # 않도록 항등 응답을 돌려준다(활성화된 rules_filter/score_and_rank/
    # 리뷰 보강이 hub 를 호출한다).
    async def filter_mobility_radius(self, origin, mobility, candidates):
        return {"filtered": candidates, "radius_m": None, "dropped": 0}

    async def score_indoor_bonus(self, pois, day_pop_max):
        return {"scored": []}

    async def fetch_reviews(self, query, *, display=3):
        return {"reviews": [], "count": 0, "query": query}


class _SeqGemini:
    def __init__(self, results: list) -> None:
        self._results = list(results)

    async def generate_structured(self, prompt, schema, *, system_instruction=None):
        return self._results.pop(0)


class _DonePublisher:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def publish(self, job_id, status, payload_json):
        self.payloads.append(json.loads(payload_json))
        return "1-0"


class _StatusPublisher:
    def __init__(self, fail: bool = False) -> None:
        self.stages: list[str] = []
        self._fail = fail

    async def publish_status(self, job_id, stage):
        if self._fail:
            raise ConnectionError("redis down")
        self.stages.append(stage)
        return "1-0"


_HAPPY_GEMINI = lambda: _SeqGemini([  # noqa: E731
    PlacesSelection(selections=[
        PlaceSelection(index=0, recommended_visit_time="오전"),
        PlaceSelection(index=1, recommended_visit_time="오후"),
    ]),
    RouteEnvelope(visit_order=[0, 1], legs=[
        Leg.model_validate({
            "from": 0, "to": 1, "mode": "walk",
            "estimated_distance_km": 1.0, "estimated_duration_min": 10,
        })
    ]),
    ReasonEnvelope(
        reasons=[
            PlaceReason(place_id=0, reason="이유0"),
            PlaceReason(place_id=1, reason="이유1"),
        ],
        clothing="겉옷",
    ),
])

_ALL_STAGES = [
    "parse_input", "fetch_weather", "search_places", "rules_filter",
    "score_and_rank", "recommend_places", "recommend_route", "llm_reason",
    "summarize_reviews", "build_payload",
]


def _invoke(req, status_pub) -> _DonePublisher:
    done = _DonePublisher()
    deps.set_hub_client(_FakeHub())
    deps.set_gemini_client(_HAPPY_GEMINI())
    deps.set_streams_publisher(done)
    if status_pub is not None:
        deps.set_status_publisher(status_pub)
    try:
        graph = build_graph()
        asyncio.run(graph.ainvoke({"job_id": "j", "request": req}))
    finally:
        deps.reset_all()
    return done


def test_happy_path_stage_sequence() -> None:
    """정상 경로에서 stage 가 그래프 순서대로 발행된다."""
    status = _StatusPublisher()
    done = _invoke(_request(), status)
    assert status.stages == _ALL_STAGES
    assert done.payloads[0]["status"] == "done"


def test_error_path_skips_stages() -> None:
    """입력 오류 시 parse_input → build_payload 만 발행된다."""
    status = _StatusPublisher()
    bad = _request(
        date=DateRange(
            date_start=date(2026, 7, 7),
            date_end=date(2026, 7, 6),
            time_start=time(9, 0),
            time_end=time(18, 0),
        )
    )
    done = _invoke(bad, status)
    assert status.stages == ["parse_input", "build_payload"]
    assert done.payloads[0]["status"] == "failed"


def test_status_failure_does_not_kill_job() -> None:
    """status 발행이 매번 실패해도 잡은 done 으로 완료된다."""
    done = _invoke(_request(), _StatusPublisher(fail=True))
    assert done.payloads[0]["status"] == "done"


def test_status_publisher_missing_is_tolerated() -> None:
    """status 발행자 미주입(단위 테스트 환경)에서도 잡이 완료된다."""
    done = _invoke(_request(), None)
    assert done.payloads[0]["status"] == "done"
