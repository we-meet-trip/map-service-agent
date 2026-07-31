"""build_graph 배선·라우팅 통합 테스트 (외부 호출 없이 Fake 주입).

핵심 계약 검증:
  - 정상 경로: 3개 LLM 노드 경유, LLM 호출 정확히 3회(≤3 예산 내),
    최종 done 페이로드에 reason/clothing 포함.
  - 입력 오류 경로: LLM 호출 0회, failed 페이로드.
  - route 실패 경로: llm_reason 건너뜀(LLM 2회), failed 페이로드.
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


def _candidate(i: int) -> dict:
    return {
        "content_id": f"c{i}",
        "source": "kakao",
        "name": f"장소{i}",
        "address": "주소",
        "lat": 37.5,
        "lng": 127.0,
    }


class _FakeHub:
    def __init__(self, places: list) -> None:
        self._places = places

    async def fetch_weather(self, province, city, date_start, date_end):
        return {"daily": [{"date": "2026-07-06", "pop": 30}]}

    async def search_places(
        self, province, city, *, mobility=None, keyword=None, size=15
    ):
        return {"places": self._places, "count": len(self._places)}

    # 룰/리뷰 엔드포인트 무해 통과 대역(A 트랙 활성 노드용). rules_filter/
    # score_and_rank/리뷰 보강이 hub 를 호출하지만 후보/순서/LLM 호출 수를
    # 바꾸지 않도록 항등 응답을 돌려준다.
    async def filter_mobility_radius(self, origin, mobility, candidates):
        return {"filtered": candidates, "radius_m": None, "dropped": 0}

    async def score_indoor_bonus(self, pois, day_pop_max):
        return {"scored": []}

    async def fetch_reviews(self, query, *, display=3):
        return {"reviews": [], "count": 0, "query": query}


class _SeqGemini:
    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls = 0

    async def generate_structured(self, prompt, schema, *, system_instruction=None):
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


def _invoke(req: AgentRequest, hub, gemini) -> tuple[dict, _CapturePublisher]:
    pub = _CapturePublisher()
    deps.set_hub_client(hub)
    deps.set_gemini_client(gemini)
    deps.set_streams_publisher(pub)
    try:
        graph = build_graph()
        out = asyncio.run(graph.ainvoke({"job_id": "job-1", "request": req}))
    finally:
        deps.reset_all()
    return out, pub


def _selection(n: int) -> PlacesSelection:
    return PlacesSelection(
        selections=[
            PlaceSelection(index=i, day=1, recommended_visit_time="오전")
            for i in range(n)
        ]
    )


def _route(n: int) -> RouteEnvelope:
    return RouteEnvelope(
        visit_order=list(range(n)),
        legs=[
            Leg.model_validate(
                {
                    "from": i,
                    "to": i + 1,
                    "mode": "walk",
                    "estimated_distance_km": 1.0,
                    "estimated_duration_min": 15,
                }
            )
            for i in range(n - 1)
        ],
    )


def _reasons(n: int) -> ReasonEnvelope:
    return ReasonEnvelope(
        reasons=[PlaceReason(place_id=i, reason=f"이유{i}") for i in range(n)],
        clothing="가벼운 겉옷",
    )


def test_happy_path_exactly_three_llm_calls() -> None:
    """정상 경로: selection→route→reason 3회 호출, done 페이로드 완성."""
    gemini = _SeqGemini([_selection(2), _route(2), _reasons(2)])
    out, pub = _invoke(
        _request(), _FakeHub([_candidate(0), _candidate(1)]), gemini
    )
    assert gemini.calls == 3
    assert out["llm_calls_used"] == 3
    assert out.get("error") is None
    payload = pub.payloads[0]
    assert payload["status"] == "done"
    assert payload["clothing"] == "가벼운 겉옷"
    assert [p["reason"] for p in payload["places"]] == ["이유0", "이유1"]
    assert payload["visit_order"] == [0, 1]


def test_parse_error_short_circuits_no_llm() -> None:
    """입력 오류: LLM 0회, 곧장 failed 페이로드."""
    gemini = _SeqGemini([])
    bad = _request(
        date=DateRange(
            date_start=date(2026, 7, 7),
            date_end=date(2026, 7, 6),  # start > end
            time_start=time(9, 0),
            time_end=time(18, 0),
        )
    )
    out, pub = _invoke(bad, _FakeHub([]), gemini)
    assert gemini.calls == 0
    payload = pub.payloads[0]
    assert payload["status"] == "failed"
    assert "date_start" in payload["error"]


def test_route_failure_skips_llm_reason() -> None:
    """route 검증 실패: llm_reason 미경유(LLM 2회), failed 페이로드."""
    # visit_order 가 순열이 아님 → recommend_route 에서 error
    bad_route = RouteEnvelope(visit_order=[0, 0], legs=[
        Leg.model_validate(
            {
                "from": 0,
                "to": 0,
                "mode": "walk",
                "estimated_distance_km": 1.0,
                "estimated_duration_min": 10,
            }
        )
    ])
    gemini = _SeqGemini([_selection(2), bad_route, _reasons(2)])
    out, pub = _invoke(
        _request(), _FakeHub([_candidate(0), _candidate(1)]), gemini
    )
    assert gemini.calls == 2  # reason 호출 없음
    payload = pub.payloads[0]
    assert payload["status"] == "failed"
    assert "permutation" in payload["error"]


def test_route_leg_connectivity_mismatch_fails() -> None:
    """legs[i] 가 visit_order[i]→[i+1] 를 잇지 않으면 실패한다.

    순열·길이는 유효하지만 구간 연결이 뒤집힌 응답(1→0)을 준다 —
    연결성 검증(agent_nodes.recommend_route)이 없으면 통과해버리는
    케이스로, 해당 검증의 회귀를 방지한다.
    """
    disconnected = RouteEnvelope(visit_order=[0, 1], legs=[
        Leg.model_validate({
            "from": 1, "to": 0,  # 역방향 — visit_order[0]→[1] 이어야 함
            "mode": "walk",
            "estimated_distance_km": 1.0,
            "estimated_duration_min": 10,
        })
    ])
    gemini = _SeqGemini([_selection(2), disconnected, _reasons(2)])
    out, pub = _invoke(
        _request(), _FakeHub([_candidate(0), _candidate(1)]), gemini
    )
    assert gemini.calls == 2  # llm_reason 미경유
    payload = pub.payloads[0]
    assert payload["status"] == "failed"
    assert "must connect" in payload["error"]


def test_reason_degrade_still_done() -> None:
    """llm_reason 실패(타임아웃)여도 잡은 done 으로 발행된다."""
    gemini = _SeqGemini([_selection(2), _route(2), asyncio.TimeoutError()])
    out, pub = _invoke(
        _request(), _FakeHub([_candidate(0), _candidate(1)]), gemini
    )
    payload = pub.payloads[0]
    assert payload["status"] == "done"
    assert payload["clothing"] is None
    assert out["degraded_reason"].startswith("llm_reason_failed:")
